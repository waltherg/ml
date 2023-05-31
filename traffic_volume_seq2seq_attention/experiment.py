import argparse
from pathlib import Path
import random

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm


# Set up device
print(f'PyTorch version: {torch.__version__}')
print(f'PyTorch built with MPS: {torch.backends.mps.is_built()}')
print(f'MPS available: {torch.backends.mps.is_available()}')
device = torch.device("cpu" if not torch.backends.mps.is_available() else "mps")
print(f'Using device: {device}')


# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_seed(42)


current_dir = Path(__file__).parent.absolute()
model_path = current_dir / 'models'


# Create model directory if it doesn't exist
if not model_path.exists():
    model_path.mkdir()


def time_encoding(df, timestamp_col):
    # Encode the day of year, day of week, and hour of day using cosine and sine functions
    df[f'{timestamp_col}_day_of_year_cos'] = np.cos(2 * np.pi * df[timestamp_col].dt.dayofyear / 365.25)
    df[f'{timestamp_col}_day_of_year_sin'] = np.sin(2 * np.pi * df[timestamp_col].dt.dayofyear / 365.25)
    df[f'{timestamp_col}_day_of_week_cos'] = np.cos(2 * np.pi * df[timestamp_col].dt.dayofweek / 7)
    df[f'{timestamp_col}_day_of_week_sin'] = np.sin(2 * np.pi * df[timestamp_col].dt.dayofweek / 7)
    df[f'{timestamp_col}_hour_of_day_cos'] = np.cos(2 * np.pi * df[timestamp_col].dt.hour / 24)
    df[f'{timestamp_col}_hour_of_day_sin'] = np.sin(2 * np.pi * df[timestamp_col].dt.hour / 24)

    return df


class DataWindow(Dataset):
    def __init__(self, data, feature_indexes, target_indexes, input_width, label_width, shift):
        self.data = data
        self.feature_indexes = feature_indexes
        self.target_indexes = target_indexes
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.window_size = input_width + label_width
        self.total_size = len(data) - self.window_size + 1

    def split_to_inputs_labels(self, window):
        inputs = window[:self.input_width, self.feature_indexes]
        labels = window[self.input_width:, self.target_indexes]
        return inputs, labels

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        window = self.data[idx:idx+self.window_size]
        inputs, labels = self.split_to_inputs_labels(window)
        return inputs, labels


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs shape: (batch_size, seq_length, hidden_size)
        # Encoder outputs are equivalent to the hidden states of the encoder LSTM
        attn_weights = torch.sum(encoder_outputs * decoder_hidden.permute(1, 0, 2), dim=2)
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(1)
        context = attn_weights.bmm(encoder_outputs)
        return context


class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(hidden_size + output_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.attention = Attention()

    def forward(self, encoder_outputs, hidden, cell, target):
        outputs = []
        for t in range(target.size(1)):  # target size is (batch_size, seq_length, output_size)
            x = target[:, t, :].unsqueeze(1)  # take the t-th time step from each sequence in the batch
            context = self.attention(encoder_outputs, hidden)
            x = torch.cat((context, x), -1)
            out, (hidden, cell) = self.lstm(x, (hidden, cell))
            out = self.fc(out.squeeze(1))
            outputs.append(out)
        outputs = torch.stack(outputs, 1)
        return outputs


class Seq2Seq(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(output_size, hidden_size)
        self.output_size = output_size

    def forward(self, source, target):
        encoder_outputs, hidden, cell = self.encoder(source)
        out = self.decoder(encoder_outputs, hidden, cell, target)
        return out

    def predict(self, source, predictions_length):
        encoder_outputs, hidden, cell = self.encoder(source)

        # initialize the first input to the decoder as zeros
        decoder_input = torch.zeros((source.size(0), 1, self.output_size), device=source.device)

        predictions = []

        for _ in range(predictions_length):
            context = self.decoder.attention(encoder_outputs, hidden)
            x = torch.cat((context, decoder_input), -1)
            out, (hidden, cell) = self.decoder.lstm(x, (hidden, cell))
            out = self.decoder.fc(out.squeeze(1))
            predictions.append(out)

            # use the current output as the next input to the decoder
            decoder_input = out.unsqueeze(1)

        return torch.cat(predictions, dim=1)


# Train the model
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    no_samples = 0

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X, y)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        no_samples += y.shape[0]

    total_loss /= no_samples

    return total_loss


# Evaluate the model
def evaluate(dataloader, model, loss_fn):
    model.eval()
    test_loss = 0
    test_mae = 0
    no_samples = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X, y)
            test_loss += loss_fn(pred, y).item()

            y = y.cpu()
            pred = pred.cpu()

            for t in range(y.shape[0]):
                test_mae += mean_absolute_error(
                    target_scaler.inverse_transform(y[t, :, :]),
                    target_scaler.inverse_transform(pred[t, :, :])
                )

            no_samples += y.shape[0]

    test_loss /= no_samples
    test_mae /= no_samples

    return test_loss, test_mae


def train_new_model(model, loss_fn, train_dataloader, val_dataloader, model_filename):

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Scheduler to decrease learning rate by a factor of 0.1 every 10 epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    # Define the number of epochs
    num_epochs = 1000

    # Define the lists to store the loss values
    train_losses = []
    val_losses = []

    # Initialize variables for early stopping
    best_val_loss = float('inf')
    best_epoch = 0
    patience = 3
    counter = 0

    # Train the model
    for epoch in tqdm(range(num_epochs)):
        # Train the model
        train_loss = train(train_dataloader, model, loss_fn, optimizer)

        # Evaluate the model
        val_loss, val_mae = evaluate(val_dataloader, model, loss_fn)
        # Store the loss values
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Print the progress
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}')

        # Check if the validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            counter = 0
            # Save the model parameters
            torch.save(model.state_dict(), model_path / f'{best_epoch}_{model_filename}')
        else:
            counter += 1

        # Stop training if the validation loss has not improved for `patience` epochs
        if counter >= patience:
            print(f'Validation loss has not improved for {patience} epochs. Stopping training.')
            break

        # Decrease the learning rate
        scheduler.step()

    # Load the model parameters from the epoch with the best validation loss
    model.load_state_dict(torch.load(model_path / f'{best_epoch}_{model_filename}'))

    # Serialize model
    torch.save(model.state_dict(), model_path / model_filename)


if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Train a traffic volume prediction model.')

    parser.add_argument(
        '--batch_size',
        type=int,
        default=512,
        help='Batch size',
    )

    # Parse the arguments
    args = parser.parse_args()
    batch_size = args.batch_size

    # Load the UCI interstate traffic data set
    df = pd.read_csv(current_dir / 'Metro_Interstate_Traffic_Volume.csv', parse_dates=['date_time'])

    # Sort the data by date_time
    df = df.sort_values('date_time')

    # Encode the time features
    df = time_encoding(df, 'date_time')

    # Define features and target
    features = [
        'temp',
        'traffic_volume',
        'date_time_day_of_year_cos',
        'date_time_day_of_year_sin',
        'date_time_day_of_week_cos',
        'date_time_day_of_week_sin',
        'date_time_hour_of_day_cos',
        'date_time_hour_of_day_sin',
    ]
    targets = ['traffic_volume']

    # Define the features to be normalized
    norm_features = [
        'temp',
    ]

    # Define targets to be normalized
    norm_targets = [
        'traffic_volume',
    ]

    # Create the data set
    df = df.set_index('date_time')
    data = df[features].copy()

    # Feature and target indexes
    feature_indexes = [data.columns.get_loc(col) for col in features]
    target_indexes = [data.columns.get_loc(col) for col in targets]

    # Define cutoff indexes for training, validation, and test sets
    train_cutoff = data.index[int(len(data) * 0.7)]
    val_cutoff = data.index[int(len(data) * 0.9)]

    # Split the data into training and validation sets
    train_data_df = data.loc[data.index < train_cutoff]
    val_data_df = data.loc[(data.index >= train_cutoff) & (data.index < val_cutoff)]
    test_data_df = data.loc[data.index >= val_cutoff]

    # Normalize the data
    feature_scaler = StandardScaler()
    train_data_df.loc[:, norm_features] = feature_scaler.fit_transform(train_data_df[norm_features])
    val_data_df.loc[:, norm_features] = feature_scaler.transform(val_data_df[norm_features])
    test_data_df.loc[:, norm_features] = feature_scaler.transform(test_data_df[norm_features])

    target_scaler = StandardScaler()
    train_data_df.loc[:, norm_targets] = target_scaler.fit_transform(train_data_df[norm_targets])
    val_data_df.loc[:, norm_targets] = target_scaler.transform(val_data_df[norm_targets])
    test_data_df.loc[:, norm_targets] = target_scaler.transform(test_data_df[norm_targets])

    # Convert the data to PyTorch tensors
    train_data = torch.tensor(train_data_df.values, dtype=torch.float32).to(device)
    val_data = torch.tensor(val_data_df.values, dtype=torch.float32).to(device)
    test_data = torch.tensor(test_data_df.values, dtype=torch.float32).to(device)

    # Define input width, label width, and shift
    input_width = 24*31
    label_width = 24*2
    shift = 1

    # Create the data windows
    train_dataset = DataWindow(
        data=train_data,
        feature_indexes=feature_indexes,
        target_indexes=target_indexes,
        input_width=input_width,
        label_width=label_width,
        shift=shift,
    )
    val_dataset = DataWindow(
        data=val_data,
        feature_indexes=feature_indexes,
        target_indexes=target_indexes,
        input_width=input_width,
        label_width=label_width,
        shift=shift,
    )
    test_dataset = DataWindow(
        data=test_data,
        feature_indexes=feature_indexes,
        target_indexes=target_indexes,
        input_width=input_width,
        label_width=label_width,
        shift=shift,
    )

    # Create the training and validation data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Define the model
    model_hidden_size = 128
    model_num_layers = 2
    model_dropout = 0.2

    model = Seq2Seq(
        input_size=len(features),
        hidden_size=model_hidden_size,
        output_size=len(targets),
    ).to(device)

    # Define the loss function
    loss_fn = nn.MSELoss()

    # Attempt loading model from file, otherwise train a new model
    model_name = f'{model.__class__.__name__}_{input_width}_{label_width}_{shift}_{batch_size}_{model_hidden_size}_{model_num_layers}_{model_dropout}'
    model_filename = f'{model_name}.pth'

    try:
        model.load_state_dict(torch.load(model_path / model_filename))
        print(f'Loaded model from file "{model_path / model_filename}"')
    except FileNotFoundError:
        print(f'No model file "{model_path / model_filename}" found. Training a new model.')
        train_new_model(model, loss_fn, train_dataloader, val_dataloader, model_filename)

    # Evaluate the model on the test data
    test_loss, test_mae = evaluate(test_dataloader, model, loss_fn)
    print(f'Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}')

    # Predict on the test data
    model.eval()

    # Create lists to store the predictions and targets
    predictions = []
    targets = []

    # Iterate over the test data
    for inputs, targets_batch in test_dataloader:
        # Move the inputs and targets to the device
        inputs = inputs.to(device)
        targets_batch = targets_batch.to(device)

        # Forward pass
        outputs = model.predict(inputs, predictions_length=label_width)
        outputs = outputs.unsqueeze(-1)

        # Move the outputs and targets to the CPU
        outputs = outputs.detach().cpu().numpy()
        targets_batch = targets_batch.detach().cpu().numpy()

        for batch_index in range(outputs.shape[0]):
            # Collect re-scaled predictions and targets
            predictions.append(target_scaler.inverse_transform(outputs[batch_index]))
            targets.append(target_scaler.inverse_transform(targets_batch[batch_index]))

    # Create dataframes of the predictions and targets
    predictions_df = pd.DataFrame(
        data=np.squeeze(np.stack(predictions), -1),
        columns=[
            offset for offset in range(1, label_width + 1)
        ],
        index=test_data_df.index[:test_dataset.total_size]
    )

    targets_df = pd.DataFrame(
        data=np.squeeze(np.stack(targets), -1),
        columns=[
            offset for offset in range(1, label_width + 1)
        ],
        index=test_data_df.index[:test_dataset.total_size]
    )

    # Set plotting style
    sns.set_style('darkgrid')
    sns.set_palette('colorblind')
    sns.set_context('paper')

    # Compute the mean absolute error per forecast horizon
    mae = (predictions_df - targets_df).abs().mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(mae.index, mae, label='MAE')
    ax.set_xlabel('Forecast Horizon / Hours')
    ax.set_ylabel('MAE / Vehicles')
    ax.set_title('Traffic Volume Prediction')
    ax.legend()
    fig.tight_layout()
    fig.savefig(current_dir / 'traffic_volume_prediction_mae.png', dpi=300)

    # Compute the mean absolute percentage error per forecast horizon
    mape = 100. * ((predictions_df - targets_df).abs() / targets_df).mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(mape.index, mape, label='MAPE')
    ax.set_xlabel('Forecast Horizon / Hours')
    ax.set_ylabel('MAPE / %')
    ax.set_title('Traffic Volume Prediction')
    ax.legend()
    fig.tight_layout()
    fig.savefig(current_dir / 'traffic_volume_prediction_mape.png', dpi=300)

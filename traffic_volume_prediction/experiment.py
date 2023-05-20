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
    def __init__(self, data, past_feature_indexes, future_feature_indexes, target_indexes, input_width, label_width, shift):
        self.data = data
        self.past_feature_indexes = past_feature_indexes
        self.future_feature_indexes = future_feature_indexes
        self.target_indexes = target_indexes
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.window_size = input_width + label_width
        self.total_size = len(data) - self.window_size + 1

    def split_to_inputs_labels(self, window):
        inputs_past = window[:self.input_width, self.past_feature_indexes]
        inputs_future = window[self.input_width:, self.future_feature_indexes]
        inputs = torch.cat((inputs_past.view(1, -1), inputs_future.view(1, -1)), dim=1)
        labels = window[self.input_width:, self.target_indexes]
        return inputs, labels

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        window = self.data[idx:idx+self.window_size]
        inputs, labels = self.split_to_inputs_labels(window)
        return inputs, labels


class TrafficPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TrafficPredictor, self).__init__()
        self.hidden_layer1 = nn.Linear(input_size, hidden_size)
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, input_seq):
        hidden_out = self.relu(self.hidden_layer1(input_seq))
        hidden_out = self.dropout(hidden_out)
        hidden_out = self.relu(self.hidden_layer2(hidden_out))
        hidden_out = self.dropout(hidden_out)
        output = self.output_layer(hidden_out)

        return output


# Train the model
def train(dataloader, model, loss_fn, optimizer, batch_size, model_input_size, model_output_size):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X.view(batch_size, model_input_size))
        loss = loss_fn(pred, y.view(batch_size, model_output_size))
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    total_loss /= size

    return total_loss


# Evaluate the model
def evaluate(dataloader, model, loss_fn, batch_size, model_input_size, model_output_size):
    size = len(dataloader.dataset)
    model.eval()
    test_loss = 0
    test_mae = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.view(batch_size, model_input_size))
            test_loss += loss_fn(pred, y.view(batch_size, model_output_size)).item()

            test_mae += mean_absolute_error(
                target_scaler.inverse_transform(y.cpu().view(batch_size, -1)),
                target_scaler.inverse_transform(pred.cpu())
            )

    test_loss /= size
    test_mae /= size

    return test_loss, test_mae


def train_new_model(model, loss_fn, train_dataloader, val_dataloader, model_filename='traffic_model.pth'):

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Define the number of epochs
    num_epochs = 1000

    # Define the lists to store the loss values
    train_losses = []
    val_losses = []

    # Initialize variables for early stopping
    best_val_loss = float('inf')
    patience = 3
    counter = 0

    # Train the model
    for epoch in tqdm(range(num_epochs)):
        # Train the model
        train_loss = train(train_dataloader, model, loss_fn, optimizer, batch_size, model_input_size, model_output_size)
        # Evaluate the model
        val_loss, val_mae = evaluate(val_dataloader, model, loss_fn, batch_size, model_input_size, model_output_size)
        # Store the loss values
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Print the progress
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}')

        # Check if the validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1

        # Stop training if the validation loss has not improved for `patience` epochs
        if counter >= patience:
            print(f'Validation loss has not improved for {patience} epochs. Stopping training.')
            break

    # Serialize model
    torch.save(model.state_dict(), current_dir / model_filename)


if __name__ == '__main__':
    # Load the UCI interstate traffic data set
    df = pd.read_csv(current_dir / 'Metro_Interstate_Traffic_Volume.csv', parse_dates=['date_time'])

    # Sort the data by date_time
    df = df.sort_values('date_time')

    # Encode the time features
    df = time_encoding(df, 'date_time')

    # Define features and target
    past_features = [
        'temp',
        'traffic_volume',
    ]
    future_features = [
        'date_time_day_of_year_cos',
        'date_time_day_of_year_sin',
        'date_time_day_of_week_cos',
        'date_time_day_of_week_sin',
        'date_time_hour_of_day_cos',
        'date_time_hour_of_day_sin',
    ]
    features = past_features + future_features
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
    past_feature_indexes = [data.columns.get_loc(col) for col in past_features]
    future_feature_indexes = [data.columns.get_loc(col) for col in future_features]
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
    input_width = 24*7
    label_width = 24*2
    shift = 1

    # Create the data windows
    train_dataset = DataWindow(
        data=train_data,
        past_feature_indexes=past_feature_indexes,
        future_feature_indexes=future_feature_indexes,
        target_indexes=target_indexes,
        input_width=input_width,
        label_width=label_width,
        shift=shift,
    )
    val_dataset = DataWindow(
        data=val_data,
        past_feature_indexes=past_feature_indexes,
        future_feature_indexes=future_feature_indexes,
        target_indexes=target_indexes,
        input_width=input_width,
        label_width=label_width,
        shift=shift,
    )
    test_dataset = DataWindow(
        data=test_data,
        past_feature_indexes=past_feature_indexes,
        future_feature_indexes=future_feature_indexes,
        target_indexes=target_indexes,
        input_width=input_width,
        label_width=label_width,
        shift=shift,
    )

    # Define batch size
    batch_size = 4096

    # Create the training and validation data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Define the model
    model_input_size = len(past_features) * input_width + len(future_features) * label_width
    model_hidden_size = 128
    model_output_size = len(targets) * label_width

    model = TrafficPredictor(
        input_size=model_input_size,
        hidden_size=model_hidden_size,
        output_size=model_output_size,
    ).to(device)

    # Define the loss function
    loss_fn = nn.MSELoss()

    # Attempt loading model from file, otherwise train a new model
    model_filename = 'traffic_model.pth'
    try:
        model.load_state_dict(torch.load(current_dir / model_filename))
        print('Loaded model from file.')
    except FileNotFoundError:
        print('No model file found. Training a new model.')
        train_new_model(model, loss_fn, train_dataloader, val_dataloader, model_filename)

    # Evaluate the model on the test data
    test_loss, test_mae = evaluate(test_dataloader, model, loss_fn, batch_size, model_input_size, model_output_size)
    print(f'Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}')

    # Predict on the test data
    model.eval()

    # Create the test data loader
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Create lists to store the predictions and targets
    predictions = []
    targets = []

    # Iterate over the test data
    for inputs, targets_batch in test_dataloader:
        # Move the inputs and targets to the device
        inputs = inputs.view(1, model_input_size).to(device)
        targets_batch = targets_batch.view(1, model_output_size).to(device)

        # Forward pass
        outputs = model(inputs)

        # Reverse the normalization
        outputs = target_scaler.inverse_transform(outputs.detach().cpu().numpy())
        targets_batch = target_scaler.inverse_transform(targets_batch.detach().cpu().numpy())

        # Append the predictions and targets
        predictions.append(outputs)
        targets.append(targets_batch)

    # Create dataframes of the predictions and targets
    predictions_df = pd.DataFrame(
        data=np.concatenate(predictions),
        columns=[
            offset for offset in range(1, label_width + 1)
        ],
        index=test_data_df.index[:test_dataset.total_size]
    )

    targets_df = pd.DataFrame(
        data=np.concatenate(targets),
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

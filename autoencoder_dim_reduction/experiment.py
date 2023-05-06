import os
from pathlib import Path
import random

import pandas as pd
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import numpy as np
import optuna
from optuna.integration.mlflow import MLflowCallback
import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)
from tqdm import tqdm
from xgboost import XGBClassifier


NO_EPOCHS = 1000


mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

experiment_name = "Autoencoder feature dimensionality reduction for XGBoost multi-class classification"

try:
    mlflow.create_experiment(experiment_name)
except mlflow.exceptions.RestException as error:
    if 'RESOURCE_ALREADY_EXISTS' in error.message:
        mlflow.set_experiment(experiment_name)

main_run = mlflow.start_run(run_name=pd.Timestamp.utcnow().isoformat())

# Check for MPS availability
print(f'PyTorch version: {torch.__version__}')
print(f'PyTorch built with MPS: {torch.backends.mps.is_built()}')
print(f'MPS available: {torch.backends.mps.is_available()}')

device = torch.device("cpu" if not torch.backends.mps.is_available() else "mps")
print(f'Using device: {device}')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_seed(42)


# Load data
base_path = Path(__file__).parent
train = pd.read_csv(base_path / 'train.csv')
test = pd.read_csv(base_path / 'test.csv')

# Preprocess data
X_train = train.drop(['subject', 'Activity'], axis=1).values
X_test = test.drop(['subject', 'Activity'], axis=1).values

y_train = train['Activity']
y_test = test['Activity']

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)


# Create autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_size, latent_dim=2):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(),
            nn.Linear(input_size // 4, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_size // 4),
            nn.ReLU(),
            nn.Linear(input_size // 4, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, input_size),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def objective(trial):
    # Hyperparameters to be tuned
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    latent_dim = trial.suggest_int("latent_dim", 10, 100)

    # Create and train the autoencoder using the hyperparameters
    autoencoder = Autoencoder(input_size=X_train.shape[1], latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    # X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    for epoch in tqdm(range(NO_EPOCHS)):
        autoencoder.train()
        optimizer.zero_grad()
        outputs = autoencoder(X_train_tensor)
        loss = criterion(outputs, X_train_tensor)
        loss.backward()
        optimizer.step()

    # Evaluate the autoencoder using the loss
    autoencoder.eval()
    with torch.no_grad():
        outputs = autoencoder(X_train_tensor)
        loss = criterion(outputs, X_train_tensor)

    return loss.item()


mlflow_callback = MLflowCallback(
    tracking_uri=os.environ['MLFLOW_TRACKING_URI'],
    mlflow_kwargs={
        'run_name': 'Autoencoder hyperparameter tuning',
        "nested": True
    },
    metric_name='mse',
)

study = optuna.create_study(direction="minimize", study_name=experiment_name)
study.optimize(objective, n_trials=5, callbacks=[mlflow_callback])  # Trial 3 found to be ideal

# Get the best hyperparameters
best_params = study.best_params
print("Best hyperparameters:", best_params)

# Train the autoencoder using the best hyperparameters
autoencoder = Autoencoder(input_size=X_train.shape[1], latent_dim=best_params["latent_dim"]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=best_params["learning_rate"])

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

with mlflow.start_run(run_name='Autoencoder model', nested=True):
    mlflow.log_params(best_params)
    mlflow.log_param("epochs", NO_EPOCHS)

    for epoch in tqdm(range(NO_EPOCHS)):
        autoencoder.train()
        optimizer.zero_grad()
        outputs = autoencoder(X_train_tensor)
        loss = criterion(outputs, X_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print(f'Epoch [{epoch + 1}/{NO_EPOCHS}], Loss: {loss.item():.6f}')
            mlflow.log_metric("training_loss", loss.item(), step=epoch)

    mlflow.pytorch.log_model(autoencoder, "autoencoder")
    mlflow.log_metric("final_loss", loss.item())

# Encode data
with torch.no_grad():
    encoded_X_train = autoencoder.encoder(X_train_tensor).cpu().numpy()
    encoded_X_test = autoencoder.encoder(X_test_tensor).cpu().numpy()

with mlflow.start_run(run_name='XGBoost classifier without dimensionality reduction', nested=True):
    # Train XGBoost model
    xgb_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)

    # Evaluate model
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print('Event classification without dimensionality reduction:')
    print(f'Dimensions before reduction: {X_train.shape[1]}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    # Log parameters and metrics
    mlflow.log_params(best_params)
    mlflow.log_param("n_estimators", xgb_model.n_estimators)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("dimensions_before_reduction", X_train.shape[1])

    # Log the XGBoost model
    mlflow.sklearn.log_model(xgb_model, "xgboost_without_reduction")

    # Log the confusion matrix
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    mlflow.log_dict(np.array(cm).tolist(), "confusion_matrix.json")

    cm = ConfusionMatrixDisplay(
        cm,
        display_labels=label_encoder.classes_,
    )
    cm.plot(xticks_rotation='vertical', values_format='.2%')
    cm.figure_.savefig("confusion_matrix.png", pad_inches=.5, bbox_inches='tight')
    mlflow.log_artifact("confusion_matrix.png")

with mlflow.start_run(run_name='XGBoost classifier with dimensionality reduction using autoencoder', nested=True):
    # Train XGBoost model
    xgb_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    xgb_model.fit(encoded_X_train, y_train)

    # Evaluate model
    y_pred = xgb_model.predict(encoded_X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print('Event classification with dimensionality reduction using autoencoder:')
    print(f'Dimensions after reduction: {encoded_X_train.shape[1]}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    # Log parameters and metrics
    mlflow.log_params(best_params)
    mlflow.log_param("n_estimators", xgb_model.n_estimators)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("dimensions_after_reduction", encoded_X_train.shape[1])

    # Log the XGBoost model
    mlflow.sklearn.log_model(xgb_model, "xgboost_with_reduction")

    # Log the confusion matrix
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    mlflow.log_dict(np.array(cm).tolist(), "confusion_matrix.json")

    cm = ConfusionMatrixDisplay(
        cm,
        display_labels=label_encoder.classes_,
    )
    cm.plot(xticks_rotation='vertical', values_format='.2%')
    cm.figure_.savefig("confusion_matrix.png", pad_inches=.5, bbox_inches='tight')
    mlflow.log_artifact("confusion_matrix.png")

# Close the MLflow main run
mlflow.end_run()

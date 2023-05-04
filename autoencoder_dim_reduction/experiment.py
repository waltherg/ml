from pathlib import Path
import random

import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from xgboost import XGBClassifier


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


# Train autoencoder
autoencoder = Autoencoder(input_size=X_train.shape[1], latent_dim=X_test.shape[1]//8).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

no_epochs = 19_000
for epoch in tqdm(range(no_epochs)):
    autoencoder.train()
    optimizer.zero_grad()
    outputs = autoencoder(X_train_tensor)
    loss = criterion(outputs, X_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{no_epochs}], Loss: {loss.item():.6f}')

# Encode data
with torch.no_grad():
    encoded_X_train = autoencoder.encoder(X_train_tensor).cpu().numpy()
    encoded_X_test = autoencoder.encoder(X_test_tensor).cpu().numpy()

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

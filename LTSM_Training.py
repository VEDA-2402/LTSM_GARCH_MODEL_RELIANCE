import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

# Load your closing price CSV
df = pd.read_csv('reliance_closing_prices_2008_to_now.csv')

# Assuming the CSV has a column 'Close'
prices = df['close'].values.reshape(-1, 1)

# Normalize prices between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Create sequences: X = past 60 days, y = next day price
def create_sequences(data, seq_length=60):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 60
X, y = create_sequences(scaled_prices, seq_length)

# Convert to PyTorch tensors
X_tensors = torch.from_numpy(X).float()
y_tensors = torch.from_numpy(y).float()

# Example train-test split (80%-20%)
train_size = int(len(X_tensors) * 0.8)
X_train = X_tensors[:train_size]
y_train = y_tensors[:train_size]
X_test = X_tensors[train_size:]
y_test = y_tensors[train_size:]

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, {y_test.shape}")

#===============LTSM MODEL TRAINING=====================#

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Define the LSTM model
class LSTMStockPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMStockPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Last time step output
        out = self.fc(out)
        return out

# Hyperparameters
input_size = 1
hidden_size = 50
num_layers = 2
output_size = 1
num_epochs = 30
batch_size = 64
learning_rate = 0.001

# Create DataLoader for batching
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMStockPredictor(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_losses = []
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    avg_loss = sum(train_losses) / len(train_losses)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

# Evaluation on test set
model.eval()
with torch.no_grad():
    predictions = []
    actuals = []
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        predictions.append(outputs.cpu())
        actuals.append(targets.cpu())

    predictions = torch.cat(predictions).numpy()
    actuals = torch.cat(actuals).numpy()

print("Testing complete")

#==================================================================#

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Inverse transform predictions and actual values if scaled
predicted_prices = scaler.inverse_transform(predictions)
actual_prices = scaler.inverse_transform(actuals)

# Calculate evaluation metrics
mse = mean_squared_error(actual_prices, predicted_prices)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual_prices, predicted_prices)
print(f'MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}')

# Plot actual vs predicted
plt.figure(figsize=(12,6))
plt.plot(actual_prices, label='Actual Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction - Actual vs Predicted')
plt.legend()
plt.show()



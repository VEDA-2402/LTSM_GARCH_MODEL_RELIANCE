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

#=====================================================#

import torch

# After preprocessing (X_train, y_train, X_test, y_test tensors)
torch.save(X_train, 'X_train.pt')
torch.save(y_train, 'y_train.pt')
torch.save(X_test, 'X_test.pt')
torch.save(y_test, 'y_test.pt')
print("Preprocessed data saved.")


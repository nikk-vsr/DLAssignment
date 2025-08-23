import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess data
data_ = pd.read_csv('RELIANCE.NS.csv', index_col='Date', parse_dates=['Date'])
data = data_.dropna(inplace=False)

input_data = data[['Open', 'High', 'Volume']]
output_data = data[['Close']].values

input_scaler = MinMaxScaler(feature_range=(0, 1))
output_scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = input_scaler.fit_transform(input_data)
scaled_output = output_scaler.fit_transform(output_data)

training_data_len = int(np.ceil(len(scaled_data) * 0.70))
train_data = scaled_data[0:training_data_len, :]
test_data = scaled_data[training_data_len:, :]

x_train, x_test, y_train, y_test, y_test_org = [], [], [], [], []
seq_length = 60

for i in range(len(train_data) - seq_length - 4):
    x_train.append(train_data[i:(i + seq_length)])
    y_train.append(scaled_output[(i + seq_length):(i + seq_length + 5)])

for i in range(len(test_data) - seq_length - 4):
    x_test.append(test_data[i:(i + seq_length)])
    y_test.append(scaled_output[training_data_len + i + seq_length:
                                training_data_len + i + seq_length + 5])
    y_test_org.append(output_data[training_data_len + i + seq_length:
                                  training_data_len + i + seq_length + 5])

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
y_test_org = np.array(y_test_org)

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)
print("y_test_org shape:", y_test_org.shape)

# Convert to torch tensors
x_train_t = torch.tensor(x_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
x_test_t = torch.tensor(x_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(x_train_t, y_train_t)
test_dataset = TensorDataset(x_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define LSTM Model
class StockLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size1=80, hidden_size2=50, output_size=5):
        super(StockLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.fc = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]  # Take last time step
        out = self.fc(out)
        out = self.relu(out)
        return out

model = StockLSTM().to(device)
print(model)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
train_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    epoch_loss /= len(train_loader.dataset)
    train_losses.append(epoch_loss)
    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}')

# Plot training loss
plt.figure(figsize=(6, 6))
plt.plot(train_losses)
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Loss'], loc='upper right')
plt.show()

# Prediction and RMSE
model.eval()
predictions_scaled = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        outputs = model(xb)
        predictions_scaled.append(outputs.cpu().numpy())
predictions_scaled = np.concatenate(predictions_scaled, axis=0)
predictions_org = output_scaler.inverse_transform(predictions_scaled)

y_test_org_flat = y_test_org.reshape(-1, 5)
rmse = math.sqrt(mean_squared_error(y_test_org_flat, predictions_org))
print("The root mean squared error is {}.".format(rmse))

# Plot predictions
def plot_predictions(test, predicted):
    plt.plot(test[:, 0], color='red', label='Real RIL Stock Price')
    plt.plot(predicted[:, 0], color='blue', label='Predicted RIL Stock Price')
    plt.title('RIL Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('RIL Stock Price')
    plt.legend()
    plt.show()

plot_predictions(y_test_org_flat, predictions_org)
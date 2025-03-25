
import matplotlib.pyplot as plt

#imports for torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# imports for the data
import kagglehub
import pandas as pd
import os
import numpy as np
import math
import yfinance as yf
import streamlit as st


# Use GPU if available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

 
df = yf.download("NVDA", start="1999-01-22", end="2025-03-15", auto_adjust=False)
df.head()

 
column_names = df.columns.get_level_values(0).tolist()
df.columns = column_names
df.head()

 
#create a column for the opening price of the next day
df['Next_Open'] = df['Open'].shift(-1)
df = df.dropna()

#plot the data
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Open'], label='Open')
plt.plot(df.index, df['Close'], label='Close')
plt.plot(df.index, df['High'], label='High')
plt.plot(df.index, df['Low'], label='Low')
plt.legend()

 
#statistical analysis of the data
df.describe()

 
#create dataset/dataloader objects
class StockDataset(torch.utils.data.Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        # Select columns corresponding to the different inputs and outputs from the dataframe we just created.
        # And convert to PyTorch tensors
        x1 = torch.tensor(self.data.iloc[idx:idx + self.window_size]['Adj Close'].values, dtype=torch.float32)
        x2 = torch.tensor(self.data.iloc[idx:idx + self.window_size]['Close'].values, dtype=torch.float32)
        x3 = torch.tensor(self.data.iloc[idx:idx + self.window_size]['High'].values, dtype=torch.float32)
        x4 = torch.tensor(self.data.iloc[idx:idx + self.window_size]['Low'].values, dtype=torch.float32)
        x5 = torch.tensor(self.data.iloc[idx:idx + self.window_size]['Open'].values, dtype=torch.float32)
        x6 = torch.tensor(self.data.iloc[idx:idx + self.window_size]['Volume'].values, dtype=torch.float32)
        y = torch.tensor(self.data.iloc[idx + self.window_size]['Next_Open'], dtype=torch.float32)
        # Assemble all input features in a single inputs tensor with 2 columns and rows for each sample in the dataset.
        inputs = torch.stack([x1, x2, x3, x4, x5, x6], dim = 0)
        return inputs, y
    

#train-test split
train_size = int(len(df) * 0.75)
test_size = len(df) - train_size
train_data = df.iloc[:train_size, :]
test_data = df.iloc[train_size:, :]
print(train_data.shape, test_data.shape)

train_dataset = StockDataset(train_data, window_size=10)
test_dataset = StockDataset(test_data, window_size=10)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

 
#model
class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size #numbers of features in input
        self.hidden_size = hidden_size #number of features in hidden state
        self.output_size = output_size
        self.num_layers = num_layers
        
        # LSTM cell
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers = self.num_layers, 
                                  batch_first = True)
        self.dropout = torch.nn.Dropout(0.2)

        
        # Linear layer for final prediction
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input_size, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.randn(self.num_layers, input_size, self.hidden_size).to(device)
            c0 = torch.randn(self.num_layers, input_size, self.hidden_size).to(device)
        
        out, (hn, cn) = self.lstm(input_size, (h0, c0))
        out = self.linear(out[:, -1, :])
        return out, hn, cn

 
# Define the model parameters
input_size = 10
hidden_size = 12
num_layers = 2
output_size = 1

#Create the model
model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
print(model)

 
#display sized of inputs and targets in train_loader
for inputs, targets in train_loader:
    print(inputs.shape, targets.shape) 
    #inputs is a tensor of size [batch_size, 6, 10] and targets is a tensor of size [batch_size]
    break

 
#training
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100

for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        model.train()
        optimizer.zero_grad()
        
        h0 = torch.randn(model.num_layers, inputs.size(0), model.hidden_size).to(device)
        c0 = torch.randn(model.num_layers, inputs.size(0), model.hidden_size).to(device)
        
        outputs, h0, c0 = model(inputs, h0, c0)
        
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        
        h0 = h0.detach()
        c0 = c0.detach()

    if (epoch+1) % 10 == 0:
        print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item()}')


 
#testing
model.eval()
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        h0 = torch.randn(model.num_layers, inputs.size(0), model.hidden_size).to(device)
        c0 = torch.randn(model.num_layers, inputs.size(0), model.hidden_size).to(device)
        
        outputs, h0, c0 = model(inputs, h0, c0)
        
        loss = criterion(outputs.squeeze(), targets)
        print(f'Test loss: {loss.item()}')

        outputs = outputs.squeeze().cpu().numpy()
        targets = targets.cpu().numpy()
        break

 
torch.save(model, 'stock_predictor.pth')
model = torch.load('stock_predictor.pth', weights_only=False)

st.title("NVIDIA Stock Prediction")
st.write("This is a simple web app to predict the opening price of NVIDIA stock for the next day.")

adj_close = st.number_input("Adj Close")
close = st.number_input("Close")
high = st.number_input("High")
low = st.number_input("Low")
open = st.number_input("Open")
volume = st.number_input("Volume")

inputs = np.array([adj_close, close, high, low, open, volume])
inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

if st.button("Predict"):
    model.eval()
    with torch.no_grad():
        outputs, _, _ = model(inputs)
        prediction = outputs.item()
        st.write(f"The predicted opening price for NVIDIA stock tomorrow is: {prediction}")
        st.write("Note: This is a simple model and the prediction may not be accurate.")




import streamlit as st
import torch
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])


# Load your trained model
def load_model():
    input_size = 10
    hidden_size = 12
    num_layers = 2
    output_size = 1

    model = LSTM(input_size, hidden_size, num_layers, output_size)
    model.load_state_dict(torch.load('model.pth', map_location='cpu'))
    model.eval()
    return model

def preprocess_data(df, window_size):
    # Example of adding extra features to make 10 total
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['EMA'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['Change'] = df['Close'].pct_change()
    df['Range'] = df['High'] - df['Low']

    df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA10', 'EMA', 'Change', 'Range']]
    df = df.dropna()
    df = df[-window_size:].values
    df = (df - df.mean(axis=0)) / df.std(axis=0)
    return torch.tensor(df, dtype=torch.float32).unsqueeze(0)

# UI
st.title('📈 NVIDIA Stock Opening Price Prediction')
model = load_model()

# User selects a date for prediction
date = st.date_input('Select date for prediction', datetime.today() + timedelta(days=1))

# Fetch latest data
data = yf.download('NVDA', start='2020-01-01', end=datetime.today())

window_size = 10

if st.button('Predict Opening Price'):
    prediction_date = datetime.today().date() + timedelta(days=1)
    days_to_predict = (date - prediction_date).days + 1

    if days_to_predict <= 0:
        st.error("Please select a future date.")
    else:
        predictions = []
        recent_data = data.copy()

        for _ in range(days_to_predict):
            if len(recent_data) < window_size:
                st.error('Not enough data for prediction.')
                break

            input_tensor = preprocess_data(recent_data, window_size)
            with torch.no_grad():
                pred = model(input_tensor).item()

            predictions.append(pred)

            # simulate next day's data (for iterative prediction)
            next_row = recent_data.iloc[-1:].copy()
            next_row['Open'] = pred
            recent_data = pd.concat([recent_data, next_row])

        if predictions:
            final_pred = predictions[-1]
            st.success(f"Predicted Opening Price on {date.strftime('%Y-%m-%d')}: ${final_pred:.2f}")

import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib
import os

# streamlit run app.py (to run)

# Set Streamlit page config
st.set_page_config(page_title="NVIDIA Stock Predictor", layout="wide")

# Define LSTM Model Class
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Title and instructions
st.title("📈 NVIDIA Stock Opening Price Prediction")
st.markdown("""
This project uses a trained **LSTM model** to predict NVIDIA's next opening price based on historical time series data.
""")
st.markdown("""50.039 Theory and Practice of Deep Learning Group 4: Abid, Radhi, Dheena""")

# Load model and scaler
try:
    model = LSTMModel()
    model.load_state_dict(torch.load("lstm_nvidia_model.pth", map_location=torch.device('cpu')))
    model.eval()
    scaler = joblib.load("scaler.save")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file with NVIDIA stock data (with 'Open' column)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Uploaded your CSV file!")
else:
    if os.path.exists("NVDA.csv"):
        df = pd.read_csv("NVDA.csv")
        st.info("Using default 'NVDA.csv' from local directory.")
    else:
        st.error("No dataset found. Please upload a file or ensure NVDA.csv exists in the directory.")
        st.stop()

# Preprocess data
try:
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df = df.dropna(subset=['Open'])  # Ensure no missing values
except Exception as e:
    st.error(f"Error processing the CSV file: {e}")
    st.stop()

# Predict next open
def prepare_input(seq_length=10):
    data = df['Open'].values[-seq_length:]
    scaled = scaler.transform(data.reshape(-1, 1))
    input_seq = torch.tensor(scaled.reshape(1, seq_length, 1), dtype=torch.float32)
    return input_seq

def predict_next_open():
    input_seq = prepare_input()
    with torch.no_grad():
        output = model(input_seq)
        return scaler.inverse_transform(output.numpy())[0][0]

# Predict button
if st.button("📊 Predict Next Opening Price"):
    try:
        prediction = predict_next_open()
        st.success(f"📈 Predicted Next Opening Price: **${prediction:.2f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Plot last 50 predicted vs actual
st.subheader("📉 Last 50 Days: Predicted vs Actual Opening Prices")

try:
    seq_length = 10
    data = df['Open'].values
    scaled_data = scaler.transform(data.reshape(-1, 1))

    X = []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i+seq_length])

    X_tensor = torch.tensor(X, dtype=torch.float32)

    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(-50, 0):
            out = model(X_tensor[i].unsqueeze(0))
            pred = scaler.inverse_transform(out.numpy())[0][0]
            preds.append(pred)

    actuals = df['Open'].values[-50:]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(actuals, label='Actual Price', color='green')
    plt.plot(preds, label='Predicted Price', color='red')
    plt.xlabel("Time Steps")
    plt.ylabel("Open Price")
    plt.title("Actual vs Predicted Open Price (Last 50 Days)")
    plt.legend()
    st.pyplot(plt)
    plt.clf()
except Exception as e:
    st.error(f"Error plotting predictions: {e}")

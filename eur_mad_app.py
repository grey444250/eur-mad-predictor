# eur_mad_app.py
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime, timedelta
import requests
import plotly.express as px

st.set_page_config(
    page_title="EUR/MAD 5-Day Forecast",
    page_icon="💶",
    layout="wide"
)

st.title("💶 EUR/MAD 5-Day Forecast")
st.write("Predictions for the next 5 business days based on the latest exchange rate.")

# Sidebar
st.sidebar.header("Options")
st.sidebar.write("Data updates daily using the latest EUR/MAD rate.")

# Load saved model and scaler
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = LSTMModel()
model.load_state_dict(torch.load("eur_mad_model.pth"))
model.eval()
scaler = joblib.load("eur_mad_scaler.pkl")

# Fetch latest rate
base_currency = "EUR"
target_currency = "MAD"
url = f"https://open.er-api.com/v6/latest/{base_currency}"
data = requests.get(url).json()
if data["result"] != "success":
    st.error("Error fetching latest rate")
    st.stop()
latest_rate = data["rates"][target_currency]

# Generate last sequence from latest rate
seq_length = 90
last_sequence = np.full((1, seq_length, 1), latest_rate)
future_days = 5
future_preds_scaled = []

with torch.no_grad():
    for _ in range(future_days):
        current_input = torch.tensor(last_sequence, dtype=torch.float32)
        pred = model(current_input).numpy()
        future_preds_scaled.append(pred[0,0])
        last_sequence = np.roll(last_sequence, -1, axis=1)
        last_sequence[0,-1,0] = pred[0,0]

# Inverse scale
future_preds = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1,1))
today = datetime.today()
future_dates = [today + timedelta(days=i+1) for i in range(future_days)]
pred_df = pd.DataFrame({"Date": future_dates, "Predicted EUR/MAD": future_preds.flatten()})
pred_df.set_index("Date", inplace=True)

# Display table and metric
st.subheader("Next 5-Day Predictions")
st.table(pred_df)
st.metric("Next Day Prediction", f"{pred_df.iloc[0,0]:.3f}")

# Plot interactive chart
fig = px.line(pred_df, y="Predicted EUR/MAD", title="EUR/MAD 5-Day Forecast")
st.plotly_chart(fig, use_container_width=True)





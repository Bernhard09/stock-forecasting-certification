import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import naive_forecast, arima_forecast

st.title("Stock Price Forecasting")

# load data
df = pd.read_csv("data.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)
price = df["Close"]

# user input
model_choice = st.selectbox(
    "Choose model",
    ["Naive", "ARIMA"]
)

horizon = st.slider(
    "Forecast horizon (days)",
    min_value=5,
    max_value=30,
    value=10
)

# forecasting
if model_choice == "Naive":
    forecast = naive_forecast(price, horizon)
else:
    forecast = arima_forecast(price, horizon)

# plot
fig, ax = plt.subplots()
ax.plot(price[-100:], label="Actual")
ax.plot(
    pd.date_range(price.index[-1], periods=horizon+1)[1:],
    forecast,
    label="Forecast"
)
ax.legend()

st.pyplot(fig)

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def naive_forecast(price, horizon):
    last_price = price.iloc[-1]
    last_diff = price.diff().iloc[-1]
    return last_price + np.cumsum([last_diff]*horizon)

def arima_forecast(price, horizon):
    diff = price.diff().dropna()
    model = ARIMA(diff, order=(1,0,0))
    model_fit = model.fit()

    diff_pred = model_fit.forecast(steps=horizon)
    last_price = price.iloc[-1]

    return last_price + np.cumsum(diff_pred.values)

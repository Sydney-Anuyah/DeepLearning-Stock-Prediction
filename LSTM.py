#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.layers import LSTM, Dropout
from tensorflow.keras.models import Sequential
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.preprocessing import MinMaxScaler
from datetime import date, timedelta


# In[10]:

# In[9]:


def RNN(ndays):
    tickers = ["^GSPC"] # List of stock tickers
    start_date = "2010-01-01" # Start date of historical data
    end_date = (date.today()-timedelta(days=ndays)).isoformat() # End date of historical data

    data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"] # Download adjusted close price data for the specified tickers and time period

    returns = data.pct_change().dropna() # Calculate daily returns and drop the first row containing NaN values
    
    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    returns_scaled = scaler.fit_transform(np.array(returns).reshape(-1,1))

    # Define the training data and labels
    X_train = []
    y_train = []
    for i in range(350, len(returns_scaled)):
        X_train.append(returns_scaled[i-350:i, 0])
        y_train.append(returns_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshape the data for input into the LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Define the LSTM architecture
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    hist = model.fit(X_train, y_train, epochs=20, batch_size=32)
    
    # Create a copy of the returns scaled data
    scaled_data = returns_scaled.copy()

    # Generate predictions for the next n_days
    for i in range(ndays):
        # Take the most recent 350 days of returns
        X_test = scaled_data[-350:]
    
        # Reshape the data for input into the LSTM
        X_test = np.reshape(X_test, (1, X_test.shape[0], 1))
    
        # Generate a prediction
        y_pred = model.predict(X_test)
    
        # Append the prediction to the scaled data
        scaled_data = np.append(scaled_data, y_pred)

    # Invert the scaling to get the predicted returns
    predicted_returns = scaler.inverse_transform(scaled_data[-ndays:].reshape(-1, 1))
    
    tickers = ["^GSPC"] # List of stock tickers
    start_date = (date.today()-timedelta(days=ndays)).isoformat()
    end_date = date.today().isoformat()

    data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"] # Download adjusted close price data for the specified tickers and time period

    returns_ = data.pct_change().dropna() # Calculate daily returns and drop the first row containing NaN values
    
    buy_sell_hold = []

    for value in returns_[-ndays:]:
        if value > np.percentile(list(returns[-350:]), 75):
            buy_sell_hold.append("Buy")
        elif value < np.percentile(list(returns[-350:]), 25):
            buy_sell_hold.append("Sell")
        else:
            buy_sell_hold.append("Hold")

    buy_sell_hold1 = []

    for value in predicted_returns[:len(buy_sell_hold)]:
        if value > np.percentile(list(returns[-350:]), 75):
            buy_sell_hold1.append("Buy")
        elif value < np.percentile(list(returns[-350:]), 25):
            buy_sell_hold1.append("Sell")
        else:
            buy_sell_hold1.append("Hold")

    accuracy = accuracy_score(buy_sell_hold, buy_sell_hold1)
    print("Accuracy:", accuracy)
    return buy_sell_hold1, accuracy
    




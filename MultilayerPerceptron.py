import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from datetime import date, timedelta


# In[22]:


def MultilayerPerceptron(ndays):
    tickers = ["^GSPC"] # List of stock tickers
    start_date = "2010-01-01" # Start date of historical data
    end_date = (date.today()-timedelta(days=ndays)).isoformat() # End date of historical data
    data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"] # Download adjusted close price data for the specified tickers and time period
    returns = data.pct_change().dropna() # Calculate daily returns and drop the first row containing NaN values
    # Define the input and target variables
    X = returns.iloc[:-1].values # Use all tickers' returns except the last row as input
    y = returns.iloc[1:].values # Use the SP500 returns shifted by 1 day as target
    
    # Split the data into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    X_train = X_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # Define and train the MLP model
    model = MLPRegressor(hidden_layer_sizes=(50, 50), activation='relu', solver='adam', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Get the last observed return value
    last_observed_return = returns.values[-1].reshape(-1, 1)

    predictions = []
    # Create a loop to forecast for the next 20 days
    for i in range(20):
        # Use the last observed return value to make a prediction for the next day
        prediction = model.predict(last_observed_return.reshape(1, -1))
    
        # Append the prediction to your last observed return value
        last_observed_return = np.append(last_observed_return[1:], prediction).reshape(-1, 1)
    
        predictions.append(prediction)
        
    tickers = ["^GSPC"] # List of stock tickers
    start_date = (date.today()-timedelta(days=ndays)).isoformat()
    end_date = date.today().isoformat()

    data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"] # Download adjusted close price data for the specified tickers and time period

    returns_ = data.pct_change().dropna() # Calculate daily returns and drop the first row containing NaN values

    buy_sell_hold = []

    for value in returns_[-ndays:]:
        if value > np.percentile(list(returns[-350:]), 70):
            buy_sell_hold.append("Buy")
        elif value < np.percentile(list(returns[-350:]), 30):
            buy_sell_hold.append("Sell")
        else:
            buy_sell_hold.append("Hold")

    buy_sell_hold1 = []

    for value in predictions[:len(buy_sell_hold)]:
        if value > np.percentile(list(returns[-350:]), 70):
            buy_sell_hold1.append("Buy")
        elif value < np.percentile(list(returns[-350:]), 30):
            buy_sell_hold1.append("Sell")
        else:
            buy_sell_hold1.append("Hold")

    accuracy = accuracy_score(buy_sell_hold, buy_sell_hold1)
    print("Accuracy:", accuracy)
    return buy_sell_hold1, accuracy







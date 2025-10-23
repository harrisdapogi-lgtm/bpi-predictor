# ============================
# BPI 5-Day Ahead Predictor with Performance Metrics
# ============================
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import datetime
import os

# def download_data(ticker, start, end):
#    return yf.download(ticker, start=start, end=end)

data = fetch_bpi_data()
print(f"Fetched {len(data)} daily rows from Alpha Vantage")

import os
import requests
import pandas as pd

# ------------------------------------------------------------------
# Fetch BPI data from Alpha Vantage
# ------------------------------------------------------------------
def fetch_bpi_data():
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError("ALPHAVANTAGE_API_KEY not set. Add it as a GitHub secret.")

    symbol = "BPI.PSE"        # BPI on the Philippine Stock Exchange
    url = (
        "https://www.alphavantage.co/query?"
        f"function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}"
        f"&outputsize=full&apikey={api_key}"
    )

    print("Fetching BPI data from Alpha Vantage…")
    r = requests.get(url, timeout=30)
    data = r.json().get("Time Series (Daily)", {})
    if not data:
        raise ValueError(f"No data returned from Alpha Vantage: {r.text[:200]}")

    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df.rename(columns={"5. adjusted close": "Adj Close"}, inplace=True)
    return df

def add_indicators(df):
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df.dropna(inplace=True)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_data(df, look_back=90, horizon=5):
    features = ['Close', 'Volume', 'MA20', 'MA50', 'RSI', 'MACD']
    scaler = MinMaxScaler((0,1))
    scaled = scaler.fit_transform(df[features])
    X, y = [], []
    for i in range(look_back, len(scaled)-horizon):
        X.append(scaled[i-look_back:i])
        y.append(scaled[i:i+horizon, 0])
    return np.array(X), np.array(y), scaler

def build_model(input_shape, horizon=5):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(horizon)
    ])
    model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
    return model

def predict_next_days(model, scaler, df, look_back=90, horizon=5):
    features = ['Close', 'Volume', 'MA20', 'MA50', 'RSI', 'MACD']
    last = df[features].tail(look_back).values
    scaled_last = scaler.transform(last)
    X_test = scaled_last.reshape(1, look_back, len(features))
    preds_scaled = model.predict(X_test)
    close_min, close_max = scaler.data_min_[0], scaler.data_max_[0]
    preds = preds_scaled[0] * (close_max - close_min) + close_min
    future_dates = pd.bdate_range(df.index[-1], periods=horizon+1, freq='B')[1:]
    return list(zip(future_dates.strftime("%Y-%m-%d"), preds))

def log_predictions(ticker, preds):
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", "predictions_log.csv")
    if not os.path.exists(log_path):
        pd.DataFrame(columns=["Date","Ticker","Predicted","Actual"]).to_csv(log_path, index=False)
    df = pd.read_csv(log_path)
    for date, val in preds:
        df = pd.concat([df, pd.DataFrame([[date,ticker,val,None]],columns=df.columns)], ignore_index=True)
    df.drop_duplicates(subset=["Date","Ticker"], keep="last", inplace=True)
    df.to_csv(log_path, index=False)
    print(f"Logged {len(preds)} new predictions.")

def update_actuals(ticker):
    log_path = os.path.join("logs","predictions_log.csv")
    if not os.path.exists(log_path): return
    df = pd.read_csv(log_path)
    start = (pd.to_datetime(df['Date']).min() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    end = datetime.date.today()
    hist = download_data(ticker, start, end)
    for i, row in df.iterrows():
        if pd.isna(row['Actual']) and row['Date'] in hist.index.strftime("%Y-%m-%d"):
            df.at[i, 'Actual'] = hist.loc[row['Date'], 'Close']
    df.to_csv(log_path, index=False)
    print("Updated actuals.")

def compute_metrics():
    """Compute daily MAPE and RMSE."""
    path = os.path.join("logs","predictions_log.csv")
    if not os.path.exists(path): return
    df = pd.read_csv(path).dropna()
    if df.empty: return
    df['Error'] = df['Actual'] - df['Predicted']
    df['APE'] = abs(df['Error']) / df['Actual'] * 100
    df_metrics = df.groupby('Date').agg({'APE':'mean','Error':'mean'}).reset_index()
    df_metrics.rename(columns={'APE':'MAPE','Error':'AvgError'}, inplace=True)
    df_metrics['RMSE'] = np.sqrt(df['Error']**2).mean()

    metrics_path = os.path.join("logs","metrics_log.csv")
    df_metrics.to_csv(metrics_path, index=False)
    print("Metrics updated.")

def main():
    ticker = "BPI"
    look_back, horizon = 90, 5
    end = datetime.date.today()
    start = end - datetime.timedelta(days=730)

    print("Downloading data...")
    df = add_indicators(download_data(ticker, start, end))
    X, y, scaler = prepare_data(df, look_back, horizon)
    model = build_model((X.shape[1], X.shape[2]), horizon)
    print("Training model...")
    model.fit(X, y, batch_size=32, epochs=15, verbose=1)
    preds = predict_next_days(model, scaler, df, look_back, horizon)
    log_predictions(ticker, preds)
    update_actuals(ticker)
    compute_metrics()

if __name__ == "__main__":
    main()


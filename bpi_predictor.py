"""
BPI Stock Predictor – Alpha Vantage Edition with Auto-Retry
Predicts next 5 days of BPI.PSE closing prices
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")


# -------------------------------------------------------------------
# Fetch BPI data from Alpha Vantage with retry & rate-limit handling
# -------------------------------------------------------------------
def fetch_bpi_data(max_retries=5):
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError("❌ ALPHAVANTAGE_API_KEY not found. Add it as a GitHub secret.")

    symbol = "BPI.PSE"  # Bank of the Philippine Islands (PSE)
    url = (
        "https://www.alphavantage.co/query?"
        f"function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}"
        f"&outputsize=full&apikey={api_key}"
    )

    for attempt in range(1, max_retries + 1):
        print(f"📡 Attempt {attempt}/{max_retries} fetching BPI data…")
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                print(f"⚠️ HTTP {r.status_code} from API, retrying…")
                raise requests.RequestException()

            json_data = r.json()
            if "Note" in json_data:
                # Alpha Vantage rate-limit message
                print("⏳ Rate limit reached. Waiting 60 s before retrying…")
                time.sleep(60)
                continue

            data = json_data.get("Time Series (Daily)", {})
            if not data:
                print("⚠️ Empty data returned, retrying in 20 s…")
                time.sleep(20)
                continue

            df = pd.DataFrame.from_dict(data, orient="index")
            df = df.astype(float)
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            df.rename(columns={"5. adjusted close": "Adj Close"}, inplace=True)
            print(f"✅ Successfully fetched {len(df)} daily rows.")
            return df

        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            if attempt < max_retries:
                wait = 10 * attempt
                print(f"🔁 Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise ValueError("Failed to fetch data after multiple attempts.") from e


# -------------------------------------------------------------------
# Prepare data for supervised learning
# -------------------------------------------------------------------
def prepare_data(data, lookback=30, horizon=5):
    values = data["Adj Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    X, y = [], []
    for i in range(len(scaled) - lookback - horizon):
        X.append(scaled[i : i + lookback])
        y.append(scaled[i + lookback : i + lookback + horizon].flatten())
    X, y = np.array(X), np.array(y)

    print(f"🧮 Dataset shapes → X: {X.shape}, y: {y.shape}")
    return X, y, scaler


# -------------------------------------------------------------------
# Build LSTM model
# -------------------------------------------------------------------
def build_model(input_shape, horizon):
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=input_shape),
        Dropout(0.2),
        Dense(horizon)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------
def main():
    print("🚀 Starting BPI Predictor run at", datetime.utcnow())

    # Fetch data with retry
    data = fetch_bpi_data()
    if data is None or len(data) < 50:
        raise ValueError("❌ Not enough data retrieved to train the model.")

    lookback, horizon = 30, 5
    X, y, scaler = prepare_data(data, lookback, horizon)
    if len(X.shape) < 3 or X.shape[0] == 0:
        raise ValueError("❌ Not enough data after windowing. Try reducing lookback.")

    # Build & train model
    model = build_model((X.shape[1], X.shape[2]), horizon)
    print("🧠 Training model...")
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)

    # Predict next 5 days
    last_window = X[-1].reshape(1, X.shape[1], X.shape[2])
    predicted_scaled = model.predict(last_window)
    predicted = scaler.inverse_transform(predicted_scaled.reshape(-1, 1)).flatten()

    future_dates = [data.index[-1] + timedelta(days=i+1) for i in range(horizon)]
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Close": predicted
    })

    # Save outputs
    os.makedirs("logs", exist_ok=True)
    forecast_path = f"logs/prediction_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv"
    forecast_df.to_csv(forecast_path, index=False)
    print(f"📁 Predictions saved to {forecast_path}")

    # Save recent data for context
    latest_data_path = "logs/latest_data.csv"
    data.tail(60)[["Adj Close"]].to_csv(latest_data_path)
    print("📈 Saved latest 60 days of data.")

    print("✅ Done.")


if __name__ == "__main__":
    main()

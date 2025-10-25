import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")


# -------------------------------------------------------------------
# Load BPI data from local CSV
# -------------------------------------------------------------------
def fetch_bpi_data():
    path = "data/bpi_stock.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Missing {path}. Please upload your CSV file.")

    print(f"📁 Loading dataset from {path}...")
    df = pd.read_csv(path, parse_dates=["Date"])
    df.set_index("Date", inplace=True)

    # Use “Close” if “Adj Close” doesn’t exist
    if "Adj Close" in df.columns:
        df.rename(columns={"Adj Close": "Adj_Close"}, inplace=True)
    elif "Close" in df.columns:
        df.rename(columns={"Close": "Adj_Close"}, inplace=True)
    else:
        raise ValueError("❌ CSV must have a 'Close' or 'Adj Close' column.")

    df.sort_index(inplace=True)
    print(f"✅ Loaded {len(df)} rows of BPI data from CSV.")
    return df


# -------------------------------------------------------------------
# (Rest of your existing functions)
# -------------------------------------------------------------------
def prepare_data(data, lookback=30, horizon=5):
    values = data["Adj_Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)
    X, y = [], []
    for i in range(len(scaled) - lookback - horizon):
        X.append(scaled[i : i + lookback])
        y.append(scaled[i + lookback : i + lookback + horizon].flatten())
    X, y = np.array(X), np.array(y)
    print(f"🧮 Dataset shapes → X: {X.shape}, y: {y.shape}")
    return X, y, scaler


def build_model(input_shape, horizon):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=False),
        Dropout(0.2),
        Dense(horizon)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def main():
    print("🚀 Starting BPI Predictor run at", datetime.utcnow())
    data = fetch_bpi_data()

    lookback, horizon = 30, 5
    X, y, scaler = prepare_data(data, lookback, horizon)

    model = build_model((X.shape[1], X.shape[2]), horizon)
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)

    last_window = X[-1].reshape(1, X.shape[1], X.shape[2])
    predicted_scaled = model.predict(last_window)
    predicted = scaler.inverse_transform(predicted_scaled.reshape(-1, 1)).flatten()

    future_dates = [data.index[-1] + timedelta(days=i+1) for i in range(horizon)]
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted": predicted
    })

    os.makedirs("logs", exist_ok=True)
    forecast_path = f"logs/prediction_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv"
    forecast_df.to_csv(forecast_path, index=False)
    print(f"✅ Saved prediction → {forecast_path}")


if __name__ == "__main__":
    main()


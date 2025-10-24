# 📊 BPI Stock Predictor Visualization + Accuracy Metrics
# Run this notebook after you have several logs/prediction_*.csv files.

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# --- 1️⃣ Load all daily prediction CSVs ---
log_files = sorted(glob.glob("logs/prediction_*.csv"))
if not log_files:
    raise FileNotFoundError("No prediction CSVs found in logs/ — run the workflow first!")

dfs = []
for f in log_files:
    df = pd.read_csv(f, parse_dates=["Date"])
    df["Source"] = os.path.basename(f)
    dfs.append(df)

pred_all = pd.concat(dfs, ignore_index=True).sort_values("Date")
print(f"Loaded {len(log_files)} prediction files, {len(pred_all)} total rows")

# --- 2️⃣ Load actual BPI data (same CSV used in training) ---
actual = pd.read_csv("data/bpi_stock.csv", parse_dates=["Date"])
actual = actual.rename(columns={"Close": "Actual_Close"})
actual = actual[["Date", "Actual_Close"]]

# --- 3️⃣ Merge actual & predicted ---
merged = pd.merge(actual, pred_all, on="Date", how="inner").sort_values("Date")

# --- 4️⃣ Calculate metrics (MAE, RMSE) ---
merged_clean = merged.dropna(subset=["Actual_Close", "Predicted_Close"])
if len(merged_clean) > 0:
    mae = mean_absolute_error(merged_clean["Actual_Close"], merged_clean["Predicted_Close"])
    rmse = np.sqrt(mean_squared_error(merged_clean["Actual_Close"], merged_clean["Predicted_Close"]))
    print(f"\n📈 Accuracy Metrics (based on overlapping dates):")
    print(f"   Mean Absolute Error (MAE):  {mae:.4f}")
    print(f"   Root Mean Squared Error (RMSE): {rmse:.4f}")
else:
    print("⚠️ Not enough overlapping dates to compute metrics yet.")

# --- 5️⃣ Plot Actual vs Predicted ---
plt.figure(figsize=(12,6))
plt.plot(merged["Date"], merged["Actual_Close"], label="Actual Close", color="blue")
plt.plot(merged["Date"], merged["Predicted_Close"], label="Predicted (5-day forecast)", color="red", linestyle="--")

plt.title("BPI Stock Price — Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Price (PHP)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 6️⃣ Optional: Save chart for records ---
os.makedirs("charts", exist_ok=True)
chart_path = f"charts/actual_vs_predicted_latest.png"
plt.savefig(chart_path, dpi=150)
print(f"\n🖼️ Chart saved to {chart_path}")

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime, timedelta

# === Setup ===
log_dir = "logs"
output_dir = "analysis"
os.makedirs(output_dir, exist_ok=True)

# === Clean up old logs (>90 days) ===
now = datetime.utcnow()
cutoff_date = now - timedelta(days=90)
deleted = 0

for f in glob.glob(os.path.join(log_dir, "prediction_*.csv")):
    fname = os.path.basename(f)
    try:
        date_str = fname.split("_")[1]
        file_date = datetime.strptime(date_str, "%Y%m%d")
        if file_date < cutoff_date:
            os.remove(f)
            deleted += 1
    except Exception:
        continue

print(f"🧹 Deleted {deleted} old prediction files (>90 days).")

# === Load predictions ===
files = sorted(glob.glob(os.path.join(log_dir, "prediction_*.csv")))
if not files:
    raise ValueError("No prediction files found in logs/")

dfs = []
for f in files:
    df = pd.read_csv(f)
    df["source_file"] = os.path.basename(f)
    dfs.append(df)

data = pd.concat(dfs)
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date")

# === Compute metrics ===
data["abs_err"] = abs(data["Actual"] - data["Predicted"])
data["pct_err"] = abs(data["Actual"] - data["Predicted"]) / data["Actual"] * 100
data["MAE_7d"] = data["abs_err"].rolling(window=7, min_periods=1).mean()
data["MAPE_7d"] = data["pct_err"].rolling(window=7, min_periods=1).mean()

mae = data["abs_err"].mean()
mape = data["pct_err"].mean()

# === Plot 1: Actual vs Predicted ===
plt.figure(figsize=(10, 5))
plt.plot(data["Date"], data["Actual"], label="Actual", linewidth=2)
plt.plot(data["Date"], data["Predicted"], label="Predicted", linestyle="--", alpha=0.7)
plt.title("📊 BPI Stock — Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Closing Price (PHP)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "bpi_plot.png"))
plt.close()

# === Plot 2: Rolling MAE/MAPE ===
plt.figure(figsize=(10, 4))
plt.plot(data["Date"], data["MAE_7d"], label="7-Day MAE")
plt.plot(data["Date"], data["MAPE_7d"], label="7-Day MAPE (%)")
plt.title("📈 Rolling 7-Day Accuracy Metrics")
plt.xlabel("Date")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "accuracy_plot.png"))
plt.close()

# === HTML Table (last 10 days) ===
latest = data.tail(10).copy()
latest_html = latest[["Date", "Actual", "Predicted"]].to_html(
    index=False,
    classes="data-table",
    border=0,
    justify="center",
    float_format="%.2f"
)

# === HTML Dashboard ===
timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
html_path = os.path.join(output_dir, "index.html")
html_content = f"""
<html>
<head>
  <title>BPI Predictor Dashboard</title>
  <meta http-equiv="refresh" content="86400">
  <style>
    body {{
      font-family: 'Segoe UI', Arial, sans-serif;
      background: #f9fafb;
      color: #1f2937;
      text-align: center;
      margin: 40px;
    }}
    h1 {{
      color: #111827;
    }}
    .stats {{
      margin: 20px auto;
      background: #ffffff;
      display: inline-block;
      padding: 15px 30px;
      border-radius: 12px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.1);
      font-size: 16px;
    }}
    img {{
      border-radius: 12px;
      margin: 20px auto;
      box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }}
    .data-table {{
      margin: 0 auto;
      border-collapse: collapse;
      background: white;
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}
    .data-table th {{
      background: #2563eb;
      color: white;
      padding: 10px;
    }}
    .data-table td {{
      padding: 8px 12px;
      border-bottom: 1px solid #e5e7eb;
    }}
    .footer {{
      margin-top: 30px;
      font-size: 14px;
      color: #6b7280;
    }}
  </style>
</head>
<body>
  <h1>📈 BPI Stock Prediction Dashboard</h1>
  <p>Last updated: {timestamp}</p>

  <div class="stats">
    <strong>MAE:</strong> {mae:.2f} PHP |
    <strong>MAPE:</strong> {mape:.2f}% |
    <strong>Records:</strong> {len(data)} |
    <strong>Old logs deleted:</strong> {deleted}
  </div>

  <h2>📊 Actual vs Predicted</h2>
  <img src="bpi_plot.png" width="800">

  <h2>📉 Rolling 7-Day Accuracy</h2>
  <img src="accuracy_plot.png" width="800">

  <h2>📅 Last 10 Records</h2>
  {latest_html}

  <div class="footer">
    Generated automatically via GitHub Actions<br>
    Repository: <a href="https://github.com/harrisdapogi-lgtm/bpi-predictor">bpi-predictor</a>
  </div>
</body>
</html>
"""

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"✅ Dashboard updated → {html_path}")

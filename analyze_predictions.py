import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime

# === Setup ===
log_dir = "logs"
output_dir = "analysis"
os.makedirs(output_dir, exist_ok=True)

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

# === Plot Actual vs Predicted ===
plt.figure(figsize=(10, 5))
plt.plot(data["Date"], data["Actual"], label="Actual", linewidth=2)
plt.plot(data["Date"], data["Predicted"], label="Predicted", linestyle="--", alpha=0.7)
plt.title("📊 BPI Stock — Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Closing Price (PHP)")
plt.legend()
plt.grid(True)

timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
plot_path = os.path.join(output_dir, "bpi_plot.png")
plt.savefig(plot_path, bbox_inches="tight")

# === Summary stats ===
mae = abs(data["Actual"] - data["Predicted"]).mean()
mape = (abs(data["Actual"] - data["Predicted"]) / data["Actual"]).mean() * 100

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
    <strong>Records:</strong> {len(data)}
  </div>

  <div>
    <img src="bpi_plot.png" width="800">
  </div>

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

print(f"✅ Dashboard saved to: {html_path}")

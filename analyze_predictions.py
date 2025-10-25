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

# === Plot ===
plt.figure(figsize=(10,5))
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

# === HTML Report ===
html_path = os.path.join(output_dir, "index.html")
html_content = f"""
<html>
<head>
<title>BPI Predictor Dashboard</title>
<meta http-equiv="refresh" content="86400">
<style>
body {{ font-family: Arial; background: #fafafa; text-align:center; }}
h1 {{ color: #1f2937; }}
img {{ border-radius: 12px; margin: 20px auto; }}
</style>
</head>
<body>
<h1>📈 BPI Stock Prediction Dashboard</h1>
<p>Last updated: {timestamp}</p>
<img src="bpi_plot.png" width="800">
<p>Data from {len(data)} rows | Generated automatically via GitHub Actions</p>
</body>
</html>
"""

with open(html_path, "w") as f:
    f.write(html_content)

print(f"✅ Dashboard saved to: {html_path}")

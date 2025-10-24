import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime

# === Paths ===
log_dir = "logs"
output_dir = "analysis"
os.makedirs(output_dir, exist_ok=True)

# === Load predictions ===
files = sorted(glob.glob(os.path.join(log_dir, "prediction_*.csv")))
if len(files) < 1:
    raise ValueError("No prediction files found!")

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

# Save HTML report with embedded plot
timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
html_path = os.path.join(output_dir, f"bpi_comparison_{timestamp}.html")
plt.savefig("temp_plot.png", bbox_inches="tight")

html_content = f"""
<html>
<head><title>BPI Actual vs Predicted ({timestamp})</title></head>
<body>
<h2>📈 BPI Stock Actual vs Predicted — {timestamp}</h2>
<img src="../temp_plot.png" width="800"><br>
<p>Total Records: {len(data)}</p>
</body>
</html>
"""

with open(html_path, "w") as f:
    f.write(html_content)

os.remove("temp_plot.png")

print(f"✅ Report saved to: {html_path}")

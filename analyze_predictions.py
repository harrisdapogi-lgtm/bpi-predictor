import pandas as pd
import glob, os, datetime

print("📈 Generating BPI prediction dashboard...")

# Collect all prediction CSVs
files = sorted(glob.glob("logs/prediction_*.csv"))
if not files:
    raise SystemExit("❌ No prediction files found in logs/")

# Combine them
data = pd.concat([pd.read_csv(f) for f in files])
data = data.sort_values("Date")

# Detect what columns exist
has_actual = "Actual" in data.columns
has_predicted = "Predicted" in data.columns

if not has_predicted:
    raise SystemExit("❌ Missing 'Predicted' column in prediction logs")

# Compute rolling accuracy if Actual exists
if has_actual:
    data["abs_err"] = abs(data["Actual"] - data["Predicted"])
    accuracy = max(0, 100 - data["abs_err"].mean() / data["Actual"].mean() * 100)
else:
    accuracy = None

# Create dashboard directory
os.makedirs("dashboard", exist_ok=True)

# Generate HTML
html_path = "dashboard/index.html"
with open(html_path, "w", encoding="utf-8") as f:
    f.write("<html><head><title>BPI Predictor Dashboard</title></head><body>")
    f.write("<h1>📊 BPI Predictor Daily Dashboard</h1>")
    f.write(f"<p>Last updated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>")

    if accuracy is not None:
        f.write(f"<h2>📈 Rolling Accuracy: {accuracy:.2f}%</h2>")
    else:
        f.write("<p>⚠️ No actual prices available yet — accuracy will appear once real data is added.</p>")

    f.write("<h3>📅 Latest Predictions</h3>")
    f.write(data.tail(10).to_html(index=False))
    f.write("</body></html>")

print(f"✅ Dashboard generated at {html_path}")

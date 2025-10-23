from flask import Flask, render_template, jsonify
import pandas as pd
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def get_data():
    log_path = os.path.join(os.path.dirname(__file__), "logs/predictions_log.csv")
    if not os.path.exists(log_path): return jsonify([])
    df = pd.read_csv(log_path).tail(60)
    return jsonify(df.to_dict(orient='records'))

@app.route('/metrics')
def get_metrics():
    metrics_path = os.path.join(os.path.dirname(__file__), "logs/metrics_log.csv")
    if not os.path.exists(metrics_path): return jsonify([])
    df = pd.read_csv(metrics_path).tail(30)
    return jsonify(df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

# app.py
from flask import Flask, jsonify, request
import pandas as pd
from pathlib import Path

app = Flask(__name__)
DIR = Path(__file__).resolve().parent

# register routes via decorators (clearer)
@app.route("/")
def root():
    return jsonify({"ok": True, "msg": "server running"})

@app.route("/fetch_data", methods=["POST"])
def fetch_data():
    data = request.get_json(silent=True) or {}
    position = (data.get("position") or "").lower()
    df_type = data.get("type", "predictions")
    model = data.get("model", "xgb")
    # optional time param (unused here)
    # time = data.get("time", "seasonal")

    if not position:
        return jsonify({"error": "Position is required"}), 400
    if position not in ["rb", "qb", "wr", "te"]:
        return jsonify({"error": "Invalid position"}), 400

    parquet_path = DIR / f"data/{position}_{df_type}_{model}.parquet"
    if not parquet_path.exists():
        return jsonify({"error": f"Data file not found: {parquet_path}"}), 404

    df = pd.read_parquet(parquet_path)
    return jsonify(df.to_dict(orient="records"))

@app.route("/get_features", methods=["POST"])
def get_features():
    data = request.get_json(silent=True) or {}
    position = (data.get("position") or "").lower()
    if not position:
        return jsonify({"error": "Position is required"}), 400
    if position not in ["rb", "qb", "wr", "te"]:
        return jsonify({"error": "Invalid position"}), 400

    features_path = DIR / f"data/{position}_features.txt"

    if not features_path.exists():
        return jsonify({"error": f"Features file not found: {features_path}"}), 404

    # If parquet:
    df = pd.read_parquet(features_path)
    return jsonify(df.to_list() if hasattr(df, "to_list") else df.tolist())

if __name__ == "__main__":
    # run with: python app.py
    app.run(debug=True, host="127.0.0.1", port=5000)
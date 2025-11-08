# app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from pathlib import Path

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
DIR = Path(__file__).resolve().parent

# register routes via decorators (clearer)
@app.route("/")
def root():
    return jsonify({"ok": True, "msg": "server running"})

@app.route("/api/fetch_data", methods=["POST"])
def fetch_data():
    data = request.get_json(silent=True) or {}
    position = (data.get("position") or "").lower()
    df_type = data.get("type", "predictions")
    model = data.get("model", "xgb")
    limit = int(data.get("limit", 50))
    offset = int(data.get("offset", 0))
    
    # Validate pagination parameters
    if limit < 1 or limit > 1000:
        limit = 50
    if offset < 0:
        offset = 0

    if not position:
        return jsonify({"error": "Position is required"}), 400
    if position not in ["rb", "qb", "wr", "te"]:
        return jsonify({"error": "Invalid position"}), 400

    parquet_path = DIR / f"data/{position}_{df_type}_{model}.parquet"
    if not parquet_path.exists():
        return jsonify({"error": f"Data file not found: {parquet_path}"}), 404

    df = pd.read_parquet(parquet_path)
    total_rows = len(df)
    
    # Apply pagination
    paginated_df = df.iloc[offset:offset + limit]
    
    return jsonify({
        "data": paginated_df.to_dict(orient="records"),
        "total": total_rows,
        "offset": offset,
        "limit": limit,
        "has_more": offset + limit < total_rows
    })

@app.route("/api/get_features", methods=["POST"])
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

    # Read .txt file line by line
    try:
        with open(features_path, 'r') as f:
            features = [line.strip() for line in f.readlines() if line.strip()]
        return jsonify(features)
    except Exception as e:
        return jsonify({"error": f"Error reading features file: {str(e)}"}), 500

if __name__ == "__main__":
    # run with: python app.py
    app.run(debug=True, host="127.0.0.1", port=5000)
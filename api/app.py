# app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from pathlib import Path
import re
from serverless_wsgi import handle_request


app = Flask(__name__)
# Enable CORS with expxlicit configuration
# This automatically handles OPTIONS preflight requests for all routes
CORS(app, 
     origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://ml-vs-human-analysts-qjyln8ob3-evans-projects-20880db0.vercel.app"],
     methods=["POST", "OPTIONS"],
     allow_headers=["Content-Type"],
     supports_credentials=False)
DIR = Path(__file__).resolve().parent

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

@app.route("/api/get_error", methods=["POST"])
def get_error():
    data = request.get_json(silent=True) or {}
    position = (data.get("position") or "").lower()
    if not position:
        return jsonify({"error": "Position is required"}), 400
    if position not in ["rb", "qb", "wr", "te"]:
        return jsonify({"error": "Invalid position"}), 400

    error_path = DIR / f"data/{position}_error.txt"

    if not error_path.exists():
        return jsonify({"error": f"Error file not found: {error_path}"}), 404

    try:
        with open(error_path, 'r') as f:
            error_text = f.read()
        return jsonify({"error_text": error_text})
    except Exception as e:
        return jsonify({"error": f"Error reading error file: {str(e)}"}), 500

@app.route("/api/get_perm_importance", methods=["POST"])
def get_perm_importance():
    data = request.get_json(silent=True) or {}
    position = (data.get("position") or "").lower()
    if not position:
        return jsonify({"error": "Position is required"}), 400
    if position not in ["rb", "qb", "wr", "te"]:
        return jsonify({"error": "Invalid position"}), 400

    perm_importance_path = DIR / f"data/{position}_perm_importance.txt"

    if not perm_importance_path.exists():
        return jsonify({"error": f"Perm importance file not found: {perm_importance_path}"}), 404

    try:
        with open(perm_importance_path, 'r') as f:
            content = f.read().strip()
        
        # Parse the single-line format: feature_name + spaces + value + " +/- " + std + next_feature...
        # Format: feature_name + spaces + importance + " +/- " + std + next_feature_name (no separator)
        # Use regex to find all patterns: number + " +/- " + number
        # Then extract feature name before each pattern
        
        # Find all "number +/- number" patterns with their positions
        pattern = r'(\d+\.?\d*)\s+\+\/-\s+(\d+\.?\d*)'
        matches = list(re.finditer(pattern, content))
        
        features = []
        for i, match in enumerate(matches):
            importance = float(match.group(1))
            std = float(match.group(2))
            
            # Find the start of this match
            match_start = match.start()
            
            # Find where the previous match ended (or start of string for first match)
            prev_end = matches[i - 1].end() if i > 0 else 0
            
            # Extract the text between previous match end and current match start
            # This should contain: feature_name + spaces + (possibly part of importance if there are multiple numbers)
            text_before = content[prev_end:match_start].strip()
            
            # Actually, simpler: the text_before ends with spaces and then the importance number
            # So we can find the last sequence of non-digit, non-space characters
            feature_match = re.search(r'([a-zA-Z_/][a-zA-Z0-9_/]*)\s*$', text_before)
            if feature_match:
                feature_name = feature_match.group(1)
            else:
                # Fallback: take everything except trailing whitespace and numbers
                feature_name = re.sub(r'\s+\d+\.?\d*\s*$', '', text_before).strip()
            
            if feature_name:
                features.append({
                    "feature": feature_name,
                    "importance": importance,
                    "std": std
                })
        
        # Sort by importance (descending) and get top 15
        features.sort(key=lambda x: x["importance"], reverse=True)
        top_features = features[:15]
        
        return jsonify({"features": top_features})
    except Exception as e:
        return jsonify({"error": f"Error reading perm importance file: {str(e)}"}), 500


def handler(request, context):
    return handle_request(app, request, context)
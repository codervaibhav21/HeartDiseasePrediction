"""
CardioScan — Flask Backend
Run:  python app.py
API:  POST /predict   { age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal }
      GET  /stats     returns dataset distribution counts
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── App setup ────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static")
CORS(app)

# ── Load model ───────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "heart_model.pkl")

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

model    = bundle["model"]
scaler   = bundle["scaler"]
FEATURES = bundle["features"]   # ['age','sex','cp','trestbps','chol','fbs',
                                 #  'restecg','thalach','exang','oldpeak','slope','ca','thal']

print(f"✅ Model loaded from {MODEL_PATH}")

# ── Dataset stats (loaded once at startup) ───────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "heart_disease_dataset.csv")
_df = pd.read_csv(DATA_PATH)
STATS = {
    "total":   int(len(_df)),
    "disease": int((_df["target"] == 1).sum()),
    "healthy": int((_df["target"] == 0).sum()),
}


# ── Helper: build a human-readable response ──────────────────────────────
def build_result(prob_disease: float) -> dict:
    risk_pct = round(prob_disease * 100)
    safe_pct = 100 - risk_pct

    if prob_disease >= 0.65:
        level       = "High Risk"
        color       = "red"
        has_disease = True
    elif prob_disease >= 0.40:
        level       = "Moderate Risk"
        color       = "amber"
        has_disease = True
    else:
        level       = "Low Risk"
        color       = "green"
        has_disease = False

    return {
        "risk_pct":    risk_pct,
        "safe_pct":    safe_pct,
        "has_disease": has_disease,
        "level":       level,
        "color":       color,          # 'red' | 'amber' | 'green'
        "prob_disease": round(float(prob_disease), 4),
        "prob_healthy": round(1 - float(prob_disease), 4),
    }


# ── Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the frontend HTML."""
    return send_from_directory("static", "index.html")


@app.route("/stats", methods=["GET"])
def stats():
    """Return dataset statistics shown in the header cards."""
    return jsonify(STATS)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a JSON body with the 13 feature values and returns
    a prediction result object for the frontend to render.
    """
    data = request.get_json(force=True)

    # ── Validate all required features are present ───────────────────────
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    # ── Build input DataFrame in the exact column order the scaler expects
    try:
        row = pd.DataFrame([[float(data[f]) for f in FEATURES]], columns=FEATURES)
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input value: {e}"}), 400

    # ── Scale → predict ──────────────────────────────────────────────────
    row_scaled      = scaler.transform(row)
    prob_disease    = model.predict_proba(row_scaled)[0][1]

    return jsonify(build_result(prob_disease))


# ── Run ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

"""
api/serve.py — Loan Approval Prediction API (Flask)
====================================================
Flask server wrapping the trained loan approval pipeline.

Endpoints:
  POST /predict         — score a single application
  POST /predict/batch   — score up to 500 applications
  GET  /health          — health check + model metadata
  GET  /metrics         — CV performance metrics

Usage:
    flask --app api/serve run --host 0.0.0.0 --port 8000
    
    or:
    python api/serve.py
"""

import json
import time
import joblib
import sys
import numpy  as np
import pandas as pd
from pathlib import Path
from flask   import Flask, request, jsonify

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# ─────────────────────────────── Config ────────────────────────────────────

MODEL_PATH = Path(__file__).parent.parent / 'models' / 'loan_model.pkl'
META_PATH  = Path(__file__).parent.parent / 'models' / 'model_meta.json'

# ─────────────────────────────── Load model ────────────────────────────────

if not MODEL_PATH.exists():
    raise RuntimeError(
        f'Model not found at {MODEL_PATH}. '
        f'Run python src/train.py first.'
    )

pipeline = joblib.load(MODEL_PATH)

with open(META_PATH) as f:
    meta = json.load(f)

THRESHOLD = meta['threshold']

# ─────────────────────────────── App ───────────────────────────────────────

app = Flask(__name__)

# ─────────────────────────────── Helpers ───────────────────────────────────

def get_risk_tier(prob: float) -> str:
    if prob < 0.30: return 'HIGH_RISK'
    if prob < 0.50: return 'MODERATE_RISK'
    if prob < 0.75: return 'LOW_RISK'
    return 'VERY_LOW_RISK'

def get_confidence(prob: float) -> str:
    distance = abs(prob - THRESHOLD)
    if distance < 0.10: return 'LOW'
    if distance < 0.25: return 'MEDIUM'
    return 'HIGH'

def validate_application(data: dict) -> tuple[bool, str]:
    """Manually validate required fields — Flask has no Pydantic."""
    required = [
        'Gender', 'Married', 'Dependents', 'Education',
        'Self_Employed', 'ApplicantIncome', 'Property_Area'
    ]
    missing = [f for f in required if f not in data]
    if missing:
        return False, f'Missing required fields: {missing}'
    if data['ApplicantIncome'] <= 0:
        return False, 'ApplicantIncome must be greater than 0'
    return True, ''

def score_application(data: dict) -> dict:
    """Score a single loan application."""
    t0 = time.time()

    row = {
        'Loan_ID'           : data.get('Loan_ID', 'LIVE'),
        'Gender'            : data.get('Gender'),
        'Married'           : data.get('Married'),
        'Dependents'        : str(data.get('Dependents', '0')),
        'Education'         : data.get('Education'),
        'Self_Employed'     : data.get('Self_Employed', 'No'),
        'ApplicantIncome'   : float(data.get('ApplicantIncome', 0)),
        'CoapplicantIncome' : float(data.get('CoapplicantIncome', 0)),
        'LoanAmount'        : data.get('LoanAmount'),
        'Loan_Amount_Term'  : data.get('Loan_Amount_Term', 360.0),
        'Credit_History'    : data.get('Credit_History'),
        'Property_Area'     : data.get('Property_Area'),
    }

    df   = pd.DataFrame([row])
    prob = float(pipeline.predict_proba(df)[0, 1])
    pred = 1 if prob >= THRESHOLD else 0

    return {
        'loan_status'           : 'Y' if pred == 1 else 'N',
        'approval_probability'  : round(prob, 4),
        'rejection_probability' : round(1 - prob, 4),
        'risk_tier'             : get_risk_tier(prob),
        'confidence'            : get_confidence(prob),
        'model_version'         : meta['model_type'],
        'inference_ms'          : round((time.time() - t0) * 1000, 2),
        'loan_id'               : data.get('Loan_ID'),
    }

# ─────────────────────────────── Endpoints ─────────────────────────────────

@app.route('/predict', methods=['POST'])
def predict():
    """Score a single loan application."""
    data = request.get_json()

    if not data:
        return jsonify({'error': 'Request body must be JSON'}), 400

    valid, error = validate_application(data)
    if not valid:
        return jsonify({'error': error}), 422

    try:
        result = score_application(data)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Score up to 500 loan applications at once."""
    data = request.get_json()

    if not data or 'applications' not in data:
        return jsonify({'error': 'Request body must contain applications list'}), 400

    applications = data['applications']

    if len(applications) > 500:
        return jsonify({'error': 'Maximum 500 applications per batch'}), 400

    t0 = time.time()
    try:
        predictions = []
        for app_data in applications:
            valid, error = validate_application(app_data)
            if not valid:
                return jsonify({'error': f'Invalid application: {error}'}), 422
            predictions.append(score_application(app_data))

        approved = sum(1 for p in predictions if p['loan_status'] == 'Y')

        return jsonify({
            'predictions'  : predictions,
            'total'        : len(predictions),
            'approved'     : approved,
            'rejected'     : len(predictions) - approved,
            'inference_ms' : round((time.time() - t0) * 1000, 2),
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check — confirms model is loaded and ready."""
    return jsonify({
        'status'        : 'healthy',
        'model_type'    : meta['model_type'],
        'threshold'     : meta['threshold'],
        'training_rows' : meta['training_rows'],
        'cv_f1'         : meta['cv_f1'],
        'cv_accuracy'   : meta['cv_accuracy'],
        'cv_roc_auc'    : meta['cv_roc_auc'],
    }), 200


@app.route('/metrics', methods=['GET'])
def metrics():
    """Return cross-validated model performance metrics."""
    return jsonify({
        'model_type'       : meta['model_type'],
        'cv_folds'         : meta['cv_folds'],
        'cv_roc_auc'       : meta['cv_roc_auc'],
        'cv_avg_precision' : meta['cv_avg_precision'],
        'cv_f1'            : meta['cv_f1'],
        'cv_accuracy'      : meta['cv_accuracy'],
        'threshold'        : meta['threshold'],
        'top_features'     : meta['top_features'],
    }), 200


@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'name'      : 'Loan Approval Prediction API',
        'version'   : '1.0.0',
        'framework' : 'Flask',
        'endpoints' : ['/predict', '/predict/batch', '/health', '/metrics'],
    }), 200


# ─────────────────────────────── Run ───────────────────────────────────────

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)


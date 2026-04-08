"""
src/predict.py — Generate Predictions on test.csv
===================================================
Loads the trained pipeline and generates predictions
on the Kaggle test set — producing a submission file.

Usage:
    python src/predict.py

Output:
    data/submission.csv  — Loan_ID + Loan_Status (Y/N)
"""

import sys
import json
import joblib
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from features import load_test

# ─────────────────────────────── Config ────────────────────────────────────

TEST_PATH      = Path(__file__).parent.parent / 'data'   / 'test.csv'
MODEL_PATH     = Path(__file__).parent.parent / 'models' / 'loan_model.pkl'
META_PATH      = Path(__file__).parent.parent / 'models' / 'model_meta.json'
SUBMISSION_PATH= Path(__file__).parent.parent / 'data'   / 'submission.csv'


# ─────────────────────────────── Main ──────────────────────────────────────

def predict():
    print('=' * 58)
    print('  Loan Approval — Generate Predictions')
    print('=' * 58)

    # ── Load model ────────────────────────────────────────────────────
    print('\n► Loading model...')
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f'Model not found at {MODEL_PATH}\n'
            f'Run python src/train.py first.'
        )

    pipeline = joblib.load(MODEL_PATH)

    with open(META_PATH) as f:
        meta = json.load(f)

    print(f'  Model type  : {meta["model_type"]}')
    print(f'  Threshold   : {meta["threshold"]}')
    print(f'  CV F1       : {meta["cv_f1"]}')
    print(f'  CV Accuracy : {meta["cv_accuracy"]}')

    # ── Load test data ────────────────────────────────────────────────
    print('\n► Loading test data...')
    X_test, loan_ids = load_test(TEST_PATH)
    print(f'  Rows: {len(X_test):,}')

    # ── Generate predictions ──────────────────────────────────────────
    print('\n► Generating predictions...')
    probs  = pipeline.predict_proba(X_test)[:, 1]
    preds  = (probs >= meta['threshold']).astype(int)
    labels = ['Y' if p == 1 else 'N' for p in preds]

    print(f'  Approved : {labels.count("Y")} ({labels.count("Y")/len(labels):.1%})')
    print(f'  Rejected : {labels.count("N")} ({labels.count("N")/len(labels):.1%})')

    # ── Save submission ───────────────────────────────────────────────
    submission = pd.DataFrame({
        'Loan_ID'     : loan_ids.values,
        'Loan_Status' : labels,
    })

    SUBMISSION_PATH.parent.mkdir(exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)

    print(f'\n✓ Submission saved → {SUBMISSION_PATH}')
    print(f'\nFirst 10 predictions:')
    print(submission.head(10).to_string(index=False))

    return submission


if __name__ == '__main__':
    predict()
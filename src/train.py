"""
src/train.py — Production Training Pipeline
=============================================
Builds a full sklearn Pipeline:

    LoanFeatureEngineer → StandardScaler → RandomForestClassifier

Steps:
  1. Load raw training data
  2. Run 5-fold cross-validation for honest OOF metrics
  3. Apply fixed threshold (0.50 — balanced problem)
  4. Fit final model on full training dataset
  5. Save pipeline + metadata to models/

Random Forest was chosen after comparing all three models in model_selection.py:
  - Random Forest     F1 0.902  ROC-AUC 0.881  Accuracy 86.2%
  - Gradient Boosting F1 0.891  ROC-AUC 0.843  Accuracy 84.6%
  - Logistic Reg      F1 0.835  ROC-AUC 0.887  Accuracy 78.9%

F1 is the primary metric — loan approval has roughly symmetric costs
between false approvals and false rejections.

Usage:
    python src/train.py
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline       import Pipeline
from sklearn.preprocessing  import StandardScaler
from sklearn.ensemble       import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).parent))
from features import LoanFeatureEngineer, load_raw

# ─────────────────────────────── Config ────────────────────────────────────

DATA_PATH    = Path(__file__).parent.parent / 'data'   / 'train.csv'
MODEL_DIR    = Path(__file__).parent.parent / 'models'
MODEL_PATH   = MODEL_DIR / 'loan_model.pkl'
META_PATH    = MODEL_DIR / 'model_meta.json'

RANDOM_STATE = 42
CV_FOLDS     = 5

# Fixed threshold — 0.50 appropriate for balanced problem
# No heavy threshold tuning needed
THRESHOLD    = 0.50


# ─────────────────────────────── Helpers ───────────────────────────────────

def print_metrics(y_true, y_pred, y_prob, label=''):
    """Print a full evaluation block."""
    print(f"\n{'='*58}")
    print(f'  {label}')
    print(f"{'='*58}")
    print(classification_report(
        y_true, y_pred,
        target_names=['Rejected', 'Approved']
    ))
    print(f'ROC-AUC       : {roc_auc_score(y_true, y_prob):.4f}')
    print(f'Avg Precision : {average_precision_score(y_true, y_prob):.4f}')
    print(f'F1 (Approved) : {f1_score(y_true, y_pred):.4f}')
    print(f'Accuracy      : {accuracy_score(y_true, y_pred):.4f}')

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f'\nConfusion matrix:')
    print(f'              Pred Rejected  Pred Approved')
    print(f'  Actual Rejected   {tn:6d}        {fp:6d}')
    print(f'  Actual Approved   {fn:6d}        {tp:6d}')


# ─────────────────────────────── Main ──────────────────────────────────────

def train():
    print('=' * 58)
    print('  Loan Approval — Production Training')
    print('=' * 58)

    # ── 1. Load data ───────────────────────────────────────────────────
    print('\n► Loading data...')
    X_raw, y = load_raw(DATA_PATH)
    print(f'  Rows         : {len(X_raw):,}')
    print(f'  Approval rate: {y.mean():.2%}  ({y.sum()} approved / {len(y):,} total)')
    print(f'  Class split  : {(y==0).sum()} rejected / {(y==1).sum()} approved')

    # ── 2. Build Pipeline ──────────────────────────────────────────────
    # Three steps chained in order:
    #   LoanFeatureEngineer  raw data → 18 numeric features
    #   StandardScaler       mean=0, std=1 for every column
    #   RandomForestClassifier  ensemble of independent trees
    print('\n► Building Pipeline...')
    pipeline = Pipeline([
        ('features', LoanFeatureEngineer()),

        ('scaler',   StandardScaler()),

        ('model',    RandomForestClassifier(
            n_estimators     = 300,
            max_depth        = 8,
            min_samples_leaf = 3,
            class_weight     = 'balanced',  # handles 69/31 class split
            random_state     = RANDOM_STATE,
            n_jobs           = -1,
        )),
    ])

    print('  Steps:')
    for name, step in pipeline.steps:
        print(f'    {name:<12} → {step.__class__.__name__}')

    # ── 3. Cross-validated OOF predictions ────────────────────────────
    print(f'\n► Running {CV_FOLDS}-fold cross-validation...')
    cv = StratifiedKFold(
        n_splits     = CV_FOLDS,
        shuffle      = True,
        random_state = RANDOM_STATE,
    )

    oof_probs = cross_val_predict(
        pipeline, X_raw, y,
        cv     = cv,
        method = 'predict_proba',
        n_jobs = -1,
    )[:, 1]

    # ── 4. Apply threshold ────────────────────────────────────────────
    # 0.50 is appropriate for this balanced problem
    # Unlike fraud detection, no heavy threshold tuning needed
    oof_preds = (oof_probs >= THRESHOLD).astype(int)
    approved  = oof_preds[y.values == 1].sum()
    rejected  = (oof_preds[y.values == 0] == 0).sum()

    print(f'  Threshold        : {THRESHOLD}')
    print(f'  Correctly approved: {approved} / {y.sum()}')
    print(f'  Correctly rejected: {rejected} / {(y==0).sum()}')

    print_metrics(y, oof_preds, oof_probs, 'Cross-Validated OOF Performance')

    # ── 5. Final fit on full dataset ───────────────────────────────────
    # Retrain on ALL 614 rows — the production model should see everything
    print('\n Fitting final model on full dataset...')
    pipeline.fit(X_raw, y)

    # Feature importances
    rf            = pipeline.named_steps['model']
    eng           = pipeline.named_steps['features']
    feature_names = eng.transform(X_raw).columns.tolist()
    importances   = pd.Series(rf.feature_importances_, index=feature_names)
    top_features  = importances.nlargest(15)

    print('\nTop 15 features by importance:')
    for feat, imp in top_features.items():
        bar = '█' * int(imp * 300)
        print(f'  {feat:<30s} {imp:.4f}  {bar}')

    # ── 6. Save artefacts ─────────────────────────────────────────────
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    meta = {
        'model_type'       : 'RandomForestClassifier',
        'threshold'        : THRESHOLD,
        'feature_names'    : feature_names,
        'top_features'     : top_features.to_dict(),
        'cv_roc_auc'       : round(float(roc_auc_score(y, oof_probs)), 4),
        'cv_avg_precision' : round(float(average_precision_score(y, oof_probs)), 4),
        'cv_f1'            : round(float(f1_score(y, oof_preds)), 4),
        'cv_accuracy'      : round(float(accuracy_score(y, oof_preds)), 4),
        'training_rows'    : len(X_raw),
        'approval_rate'    : round(float(y.mean()), 4),
        'cv_folds'         : CV_FOLDS,
        'rf_params'        : {
            'n_estimators'    : 300,
            'max_depth'       : 8,
            'min_samples_leaf': 3,
            'class_weight'    : 'balanced',
        },
    }

    with open(META_PATH, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\n Pipeline saved  - {MODEL_PATH}')
    print(f' Metadata saved  - {META_PATH}')
    print(f'\n  ROC-AUC (CV)  : {meta["cv_roc_auc"]}')
    print(f'  F1 (CV)       : {meta["cv_f1"]}')
    print(f'  Accuracy (CV) : {meta["cv_accuracy"]}')
    print(f'  Threshold     : {THRESHOLD}')

    return pipeline, meta


if __name__ == '__main__':
    train()

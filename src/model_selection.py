"""
src/model_selection.py — Compare Models Before Committing to One
=================================================================
Run this BEFORE train.py.

Compares three candidate models using 5-fold cross-validation
and ranks by F1 — appropriate for loan approval where the cost
of false approvals and false rejections is roughly symmetric.

Models compared:
  1. Logistic Regression  — simple linear baseline
  2. Random Forest        — ensemble of independent trees
  3. Gradient Boosting    — sequential boosted trees

Usage:
    python src/model_selection.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from time import perf_counter

warnings.filterwarnings('ignore')

from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline       import Pipeline
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, f1_score,
    precision_score, recall_score, accuracy_score,
)

sys.path.insert(0, str(Path(__file__).parent))
from features import LoanFeatureEngineer, load_raw

# ─────────────────────────────── Config ────────────────────────────────────

DATA_PATH    = Path(__file__).parent.parent / 'data' / 'train.csv'
RANDOM_STATE = 42
CV_FOLDS     = 5
THRESHOLD    = 0.50   # balanced problem — default threshold is appropriate


# ─────────────────────────────── Helpers ───────────────────────────────────

def evaluate_model(name, pipeline, X, y, cv):
    """Run cross-validated evaluation for one model."""
    print(f'  Testing {name}...', end=' ', flush=True)
    t0 = perf_counter()

    oof_probs = cross_val_predict(
        pipeline, X, y,
        cv=cv, method='predict_proba', n_jobs=-1
    )[:, 1]

    elapsed   = perf_counter() - t0
    oof_preds = (oof_probs >= THRESHOLD).astype(int)

    return {
        'model'      : name,
        'roc_auc'    : round(roc_auc_score(y, oof_probs), 4),
        'avg_prec'   : round(average_precision_score(y, oof_probs), 4),
        'f1'         : round(f1_score(y, oof_preds), 4),
        'accuracy'   : round(accuracy_score(y, oof_preds), 4),
        'precision'  : round(precision_score(y, oof_preds, zero_division=0), 4),
        'recall'     : round(recall_score(y, oof_preds), 4),
        'threshold'  : THRESHOLD,
        'train_time' : round(elapsed, 1),
        'oof_probs'  : oof_probs,
        'oof_preds'  : oof_preds,
    }


def print_results_table(results):
    print()
    print('=' * 75)
    print(f"  {'Model':<25} {'ROC-AUC':>8} {'F1':>7} {'Accuracy':>9} {'Recall':>8} {'Prec':>7}")
    print('=' * 75)
    for r in results:
        marker = '  best ROC-AUC' if r == results[0] else ''
        print(
            f"  {r['model']:<25} "
            f"{r['roc_auc']:>8.4f} "
            f"{r['f1']:>7.4f} "
            f"{r['accuracy']:>9.4f} "
            f"{r['recall']:>8.4f} "
            f"{r['precision']:>7.4f}"
            f"{marker}"
        )
    print('=' * 75)


def print_classification_reports(results, y):
    for r in results:
        print(f"\n── {r['model']} {'─'*40}")
        print(f"   Threshold: {r['threshold']:.2f}  |  Train time: {r['train_time']}s")
        print(classification_report(
            y, r['oof_preds'],
            target_names=['Rejected', 'Approved'],
            digits=3,
        ))


# ─────────────────────────────── Main ──────────────────────────────────────

def run_model_selection():
    print('=' * 58)
    print('  Model Selection — Loan Approval Prediction')
    print('=' * 58)

    # ── Load data ─────────────────────────────────────────────────────
    print('\n► Loading data...')
    X_raw, y = load_raw(DATA_PATH)
    print(f'  Rows         : {len(X_raw):,}')
    print(f'  Approval rate: {y.mean():.2%}')
    print(f'  Class split  : {(y==1).sum()} approved / {(y==0).sum()} rejected')

    # ── Define candidate models ───────────────────────────────────────
    candidates = {
        'Logistic Regression': Pipeline([
            ('features', LoanFeatureEngineer()),
            ('scaler',   StandardScaler()),
            ('model',    LogisticRegression(
                class_weight = 'balanced',
                max_iter     = 1000,
                C            = 0.1,
                random_state = RANDOM_STATE,
                n_jobs       = -1,
            )),
        ]),

        'Random Forest': Pipeline([
            ('features', LoanFeatureEngineer()),
            ('scaler',   StandardScaler()),
            ('model',    RandomForestClassifier(
                n_estimators     = 300,
                max_depth        = 8,
                min_samples_leaf = 3,
                class_weight     = 'balanced',
                random_state     = RANDOM_STATE,
                n_jobs           = -1,
            )),
        ]),

        'Gradient Boosting': Pipeline([
            ('features', LoanFeatureEngineer()),
            ('scaler',   StandardScaler()),
            ('model',    GradientBoostingClassifier(
                n_estimators  = 200,
                max_depth     = 4,
                learning_rate = 0.05,
                subsample     = 0.8,
                random_state  = RANDOM_STATE,
            )),
        ]),
    }

    # ── Cross-validate all models ─────────────────────────────────────
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print(f'\n► Running {CV_FOLDS}-fold cross-validation on {len(candidates)} models...')
    results = []
    for name, pipeline in candidates.items():
        result = evaluate_model(name, pipeline, X_raw, y, cv)
        results.append(result)
        print(f'done ({result["train_time"]}s)')

    # Sort by F1 — appropriate for balanced loan approval problem
    results.sort(key=lambda r: r['roc_auc'], reverse=True)

    # ── Print results ─────────────────────────────────────────────────
    print_results_table(results)
    print('\n Full classification reports (OOF predictions):')
    print_classification_reports(results, y)

    # ── Winner ────────────────────────────────────────────────────────
    winner = results[0]
    print('\n' + '=' * 58)
    print(f"  Winner   : {winner['model']}")
    print(f"  ROC-AUC  : {winner['roc_auc']}  ← primary metric")
    print(f"  F1       : {winner['f1']}")
    print(f"  Accuracy : {winner['accuracy']}")
    print(f"  Threshold: {winner['threshold']}")
    print('=' * 58)
    print(f"\n  → Use {winner['model']} in train.py\n")

    return results


if __name__ == '__main__':
    run_model_selection()
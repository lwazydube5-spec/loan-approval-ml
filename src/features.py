"""
src/features.py — Loan Approval Feature Engineering
=====================================================
Contains:
  - LoanFeatureEngineer  : sklearn-compatible transformer
  - load_raw()           : load and validate raw CSV
  - KNOWN_CATEGORIES     : fixed OHE categories — prevents column mismatch
  - BINARY_MAPS          : explicit encoding maps
  - DEPENDENTS_MAP       : ordinal encoding for Dependents

Design decisions from EDA (see Loan_Prediction.ipynb):
  - Credit_History missing → category 2.0 (not imputed)
    Missing has 64% approval rate — different from good (80%) and bad (8%)
  - LoanAmount / Loan_Amount_Term → mean imputation (continuous, low skew impact)
  - Categorical → mode imputation via SimpleImputer
  - 6 engineered features capturing income, loan, and repayability signals
  - Property_Area → OHE with fixed categories to prevent column mismatch

Usage:
    from features import LoanFeatureEngineer, load_raw
"""

import numpy  as np
import pandas as pd
from pathlib import Path
from sklearn.base       import BaseEstimator, TransformerMixin
from sklearn.impute     import SimpleImputer

# ─────────────────────────────── Constants ─────────────────────────────────

BINARY_MAPS = {
    'Gender':        {'Male': 0, 'Female': 1},
    'Married':       {'No': 0,  'Yes': 1},
    'Education':     {'Not Graduate': 0, 'Graduate': 1},
    'Self_Employed': {'No': 0,  'Yes': 1},
}

DEPENDENTS_MAP = {'0': 0, '1': 1, '2': 2, '3+': 3}

# Fixed OHE categories — learned from training data
# Hardcoded so inference never produces different columns
KNOWN_CATEGORIES = {
    'Property_Area': ['Rural', 'Semiurban', 'Urban'],
}

NUMERIC_COLS = ['ApplicantIncome', 'CoapplicantIncome',
                'LoanAmount', 'Loan_Amount_Term']

CATEGORICAL_COLS = ['Gender', 'Married', 'Dependents',
                    'Education', 'Self_Employed', 'Property_Area']

DROP_COLS = ['Loan_ID']


# ─────────────────────────────── Transformer ───────────────────────────────

class LoanFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Full feature engineering pipeline for loan approval prediction.

    Steps (in order):
      1. Drop ID columns
      2. Credit_History missing → 2.0 (third category)
      3. Impute numeric columns with mean (learned from fit)
      4. Impute categorical columns with mode (learned from fit)
      5. Binary encode Gender, Married, Education, Self_Employed
      6. Ordinal encode Dependents
      7. One-hot encode Property_Area using fixed KNOWN_CATEGORIES
      8. Engineer 6 new features from income and loan columns

    sklearn-compatible: fit() learns imputation values from training data.
    transform() applies those same values to any dataset — train, val, or test.
    This prevents data leakage by ensuring test statistics never influence training.
    """

    def __init__(self):
        self.num_imputer_  = SimpleImputer(strategy='mean')
        self.cat_imputer_  = SimpleImputer(strategy='most_frequent')
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y=None):
        """
        Learn imputation values from training data only.
        Called once on training data — never on validation or test.
        """
        df = X.copy()

        # Credit_History handled separately — not standard imputation
        df['Credit_History'] = df['Credit_History'].fillna(2.0)

        # Learn numeric imputation values from train
        self.num_imputer_.fit(df[NUMERIC_COLS])

        # Learn categorical imputation values from train
        self.cat_imputer_.fit(df[CATEGORICAL_COLS])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all transformations using values learned during fit().
        Safe to call on train, validation, and test.
        """
        df = X.copy()

        # ── 1. Drop ID ──────────────────────────────────────────────────
        for col in DROP_COLS:
            if col in df.columns:
                df = df.drop(columns=[col])

        # ── 2. Credit_History — missing → third category ────────────────
        # EDA showed: missing → 64% approval rate
        # Different from good (80%) and bad (8%) — deserves its own bucket
        df['Credit_History'] = df['Credit_History'].fillna(2.0)

        # ── 3. Numeric imputation — apply learned mean ──────────────────
        df[NUMERIC_COLS] = self.num_imputer_.transform(df[NUMERIC_COLS])

        # ── 4. Categorical imputation — apply learned mode ───────────────
        df[CATEGORICAL_COLS] = self.cat_imputer_.transform(df[CATEGORICAL_COLS])

        # ── 5. Binary encoding ──────────────────────────────────────────
        for col, mapping in BINARY_MAPS.items():
            df[col] = df[col].map(mapping).fillna(0).astype(int)

        # ── 6. Ordinal encoding — Dependents ────────────────────────────
        # '3+' → 3 preserving rank order: more dependants = higher burden
        df['Dependents'] = (df['Dependents']
                              .astype(str)
                              .map(DEPENDENTS_MAP)
                              .fillna(0)
                              .astype(int))

        # ── 7. One-hot encoding — Property_Area ─────────────────────────
        # Use fixed KNOWN_CATEGORIES to guarantee identical columns
        # at training, validation, and inference time
        for cat in KNOWN_CATEGORIES['Property_Area'][1:]:  # drop_first equivalent
            df[f'Property_Area_{cat}'] = (df['Property_Area'] == cat).astype(int)
        df = df.drop(columns=['Property_Area'])

        # ── 8. Feature engineering ───────────────────────────────────────
        # All features grounded in loan repayability business logic

        # Combined household income — co-applicant dramatically helps eligibility
        df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']

        # Income per unit of loan — higher = easier to repay
        df['IncomePerLoan'] = df['TotalIncome'] / (df['LoanAmount'] + 1)

        # Log transforms — reduces right-skew and outlier effect
        df['LoanAmount_log']  = np.log1p(df['LoanAmount'])
        df['TotalIncome_log'] = np.log1p(df['TotalIncome'])

        # Has co-applicant — even small income changes risk profile
        df['Has_Coapplicant'] = (df['CoapplicantIncome'] > 0).astype(int)

        # Debt to income — standard credit risk ratio
        df['DebtToIncome'] = df['LoanAmount'] / (df['TotalIncome'] + 1)

        self.feature_names_ = df.columns.tolist()
        return df

    def get_feature_names_out(self):
        """Return feature names after transformation."""
        return self.feature_names_


# ─────────────────────────────── Data loader ───────────────────────────────

def load_raw(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load raw training CSV and return (X, y).

    Expects train.csv format — must contain Loan_Status column.
    Raises FileNotFoundError if path does not exist.
    Raises ValueError if required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)

    required = ['Loan_Status', 'Credit_History', 'ApplicantIncome',
                'CoapplicantIncome', 'LoanAmount']
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    X = df.drop(columns=['Loan_Status'])
    y = (df['Loan_Status'] == 'Y').astype(int)

    return X, y


def load_test(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load raw test CSV and return (X, loan_ids).

    Returns features and Loan_ID series for submission file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {path}")

    df = pd.read_csv(path)
    loan_ids = df['Loan_ID'].copy()
    return df, loan_ids


if __name__ == '__main__':
    # Quick smoke test
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    train_path = Path(__file__).parent.parent / 'data' / 'train.csv'
    X_raw, y   = load_raw(train_path)

    eng = LoanFeatureEngineer()
    X_transformed = eng.fit_transform(X_raw)

    print(f'Raw features    : {X_raw.shape[1]}')
    print(f'Engineered      : {X_transformed.shape[1]}')
    print(f'Training rows   : {len(X_raw)}')
    print(f'Approval rate   : {y.mean():.2%}')
    print(f'Any nulls       : {X_transformed.isnull().sum().sum()}')
    print(f'String columns  : {X_transformed.select_dtypes(include="object").shape[1]}')
    print()
    print('Features:')
    for col in X_transformed.columns:
        print(f'  {col}')
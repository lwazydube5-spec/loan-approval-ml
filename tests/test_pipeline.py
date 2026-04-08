"""
tests/test_pipeline.py — Automated tests for the loan approval pipeline
"""
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from features import LoanFeatureEngineer, load_raw

# ── Sample application for testing ───────────────────────────────────────
SAMPLE_APPLICATION = {
    'Loan_ID'           : 'LP001002',
    'Gender'            : 'Male',
    'Married'           : 'Yes',
    'Dependents'        : '0',
    'Education'         : 'Graduate',
    'Self_Employed'     : 'No',
    'ApplicantIncome'   : 5000,
    'CoapplicantIncome' : 1500.0,
    'LoanAmount'        : 128.0,
    'Loan_Amount_Term'  : 360.0,
    'Credit_History'    : 1.0,
    'Property_Area'     : 'Semiurban',
}

# Application with all missing values
MISSING_APPLICATION = {
    'Loan_ID'           : 'LP999999',
    'Gender'            : np.nan,
    'Married'           : np.nan,
    'Dependents'        : np.nan,
    'Education'         : 'Graduate',
    'Self_Employed'     : np.nan,
    'ApplicantIncome'   : 3000,
    'CoapplicantIncome' : 0.0,
    'LoanAmount'        : np.nan,
    'Loan_Amount_Term'  : np.nan,
    'Credit_History'    : np.nan,
    'Property_Area'     : 'Urban',
}


@pytest.fixture
def engineer():
    return LoanFeatureEngineer()


@pytest.fixture
def sample_df():
    return pd.DataFrame([SAMPLE_APPLICATION])


@pytest.fixture
def fitted_engineer(sample_df):
    eng = LoanFeatureEngineer()
    eng.fit(sample_df)
    return eng


def test_engineer_fit_returns_self(engineer, sample_df):
    """fit() must return self for Pipeline compatibility."""
    result = engineer.fit(sample_df)
    assert result is engineer


def test_engineer_output_is_dataframe(fitted_engineer, sample_df):
    """transform() must return a DataFrame."""
    result = fitted_engineer.transform(sample_df)
    assert isinstance(result, pd.DataFrame)


def test_engineer_drops_loan_id(fitted_engineer, sample_df):
    """Loan_ID should be dropped — it is just an identifier."""
    result = fitted_engineer.transform(sample_df)
    assert 'Loan_ID' not in result.columns


def test_engineer_no_string_columns(fitted_engineer, sample_df):
    """All columns must be numeric after transformation."""
    result = fitted_engineer.transform(sample_df)
    string_cols = result.select_dtypes(include='object').columns.tolist()
    assert len(string_cols) == 0, f'String columns remain: {string_cols}'


def test_engineer_no_nulls(fitted_engineer, sample_df):
    """No null values should remain after transformation."""
    result = fitted_engineer.transform(sample_df)
    null_cols = result.columns[result.isnull().any()].tolist()
    assert len(null_cols) == 0, f'Null columns: {null_cols}'


def test_credit_history_missing_becomes_2(fitted_engineer):
    """Missing Credit_History must become 2.0 not imputed to mode."""
    df = pd.DataFrame([{**SAMPLE_APPLICATION, 'Credit_History': np.nan}])
    result = fitted_engineer.transform(df)
    assert result['Credit_History'].iloc[0] == 2.0


def test_credit_history_good_preserved(fitted_engineer, sample_df):
    """Good Credit_History (1.0) must stay 1.0."""
    result = fitted_engineer.transform(sample_df)
    assert result['Credit_History'].iloc[0] == 1.0


def test_binary_encoding_gender(fitted_engineer):
    """Male should encode to 0."""
    df = pd.DataFrame([{**SAMPLE_APPLICATION, 'Gender': 'Male'}])
    result = fitted_engineer.transform(df)
    assert result['Gender'].iloc[0] == 0


def test_binary_encoding_education(fitted_engineer):
    """Graduate should encode to 1."""
    result = fitted_engineer.transform(pd.DataFrame([SAMPLE_APPLICATION]))
    assert result['Education'].iloc[0] == 1


def test_dependents_ordinal_encoding(fitted_engineer):
    """3+ dependents should encode to 3."""
    df = pd.DataFrame([{**SAMPLE_APPLICATION, 'Dependents': '3+'}])
    result = fitted_engineer.transform(df)
    assert result['Dependents'].iloc[0] == 3


def test_total_income_created(fitted_engineer, sample_df):
    """TotalIncome feature must be created."""
    result = fitted_engineer.transform(sample_df)
    assert 'TotalIncome' in result.columns
    expected = SAMPLE_APPLICATION['ApplicantIncome'] + SAMPLE_APPLICATION['CoapplicantIncome']
    assert result['TotalIncome'].iloc[0] == expected


def test_has_coapplicant_flag(fitted_engineer):
    """Has_Coapplicant should be 1 when CoapplicantIncome > 0."""
    df = pd.DataFrame([{**SAMPLE_APPLICATION, 'CoapplicantIncome': 1500}])
    result = fitted_engineer.transform(df)
    assert result['Has_Coapplicant'].iloc[0] == 1


def test_no_coapplicant_flag(fitted_engineer):
    """Has_Coapplicant should be 0 when CoapplicantIncome == 0."""
    df = pd.DataFrame([{**SAMPLE_APPLICATION, 'CoapplicantIncome': 0}])
    result = fitted_engineer.transform(df)
    assert result['Has_Coapplicant'].iloc[0] == 0


def test_missing_values_handled(fitted_engineer):
    """Application with all missing values should not crash."""
    df = pd.DataFrame([MISSING_APPLICATION])
    result = fitted_engineer.transform(df)
    assert result.isnull().sum().sum() == 0


def test_column_count_stable(fitted_engineer):
    """Two identical applications must produce the same column count."""
    df1 = pd.DataFrame([SAMPLE_APPLICATION])
    df2 = pd.DataFrame([SAMPLE_APPLICATION])
    r1  = fitted_engineer.transform(df1)
    r2  = fitted_engineer.transform(df2)
    assert r1.shape[1] == r2.shape[1]


def test_log_transforms_positive(fitted_engineer, sample_df):
    """Log-transformed features must be non-negative."""
    result = fitted_engineer.transform(sample_df)
    assert result['LoanAmount_log'].iloc[0] >= 0
    assert result['TotalIncome_log'].iloc[0] >= 0

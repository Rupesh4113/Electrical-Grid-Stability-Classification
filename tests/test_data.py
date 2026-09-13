"""Tests for data loading and schema validation."""

import numpy as np
import pandas as pd
import pytest

from config.config import INPUT_FEATURES, LEAKAGE_COL, TARGET_COL
from src.data_loader import load_raw_data
from src.validation import DataValidationError, validate_raw_dataset


def test_data_loader_returns_dataframe():
    """Verify raw dataset loads into a non-empty pandas DataFrame."""
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 1000
    assert TARGET_COL in df.columns


def test_validation_schema_and_leakage_removal():
    """Verify that required features are preserved and continuous stab is dropped."""
    df = load_raw_data()
    X, y, report = validate_raw_dataset(df)

    assert set(X.columns) == set(INPUT_FEATURES)
    assert LEAKAGE_COL not in X.columns
    assert report["dropped_leakage_col"] is True
    assert set(y.unique()).issubset({0, 1})
    assert len(X) == len(y)


def test_validation_rejects_missing_columns():
    """Verify validation raises error when required feature is absent."""
    df = pd.DataFrame({"tau1": [1.0], "tau2": [2.0], TARGET_COL: ["stable"]})
    with pytest.raises(DataValidationError, match="Missing required feature columns"):
        validate_raw_dataset(df, min_observations=1)


def test_validation_rejects_infinite_values():
    """Verify validation detects and raises error on infinite values."""
    df = pd.DataFrame({col: [1.0] for col in INPUT_FEATURES})
    df["tau1"] = np.inf
    df[TARGET_COL] = "stable"
    with pytest.raises(DataValidationError, match="infinite values"):
        validate_raw_dataset(df, min_observations=1)


def test_target_distribution_calculation():
    """Verify dynamic calculation of target summary percentages."""
    df = load_raw_data()
    _, y, report = validate_raw_dataset(df)

    summary = report["target_summary"]
    total_pct = sum(item["Percentage"] for item in summary)
    assert pytest.approx(total_pct, 0.1) == 100.0

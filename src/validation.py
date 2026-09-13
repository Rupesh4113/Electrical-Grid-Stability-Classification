"""Data validation and schema integrity checking module."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.config import (
    INPUT_FEATURES,
    LEAKAGE_COL,
    TARGET_COL,
    TARGET_MAPPING,
)

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Exception raised when dataset fails integrity validation checks."""
    pass


def validate_raw_dataset(
    df: pd.DataFrame,
    required_features: Optional[List[str]] = None,
    target_col: str = TARGET_COL,
    leakage_col: str = LEAKAGE_COL,
    min_observations: int = 100,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Validate dataset schema, data types, missing values, duplicates, and target classes.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    required_features : List[str], optional
        List of required input feature columns. Defaults to INPUT_FEATURES.
    target_col : str, default TARGET_COL
        Target column name in dataframe.
    leakage_col : str, default LEAKAGE_COL
        Column to drop to avoid target leakage (e.g. continuous stab root).
    min_observations : int, default 100
        Minimum expected observation count.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]
        X (validated feature dataframe), y (normalized binary target 0/1), and validation report.
    """
    if required_features is None:
        required_features = INPUT_FEATURES

    report: Dict[str, Any] = {
        "raw_shape": df.shape,
        "missing_values": {},
        "duplicate_rows": 0,
        "infinite_values": {},
        "dropped_leakage_col": False,
        "target_summary": {},
        "feature_ranges": {},
    }

    # 1. Observation count check
    if len(df) < min_observations:
        raise DataValidationError(
            f"Dataset has {len(df)} rows, expected at least {min_observations}."
        )

    # 2. Required columns check
    missing_cols = [col for col in required_features if col not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required feature columns: {missing_cols}")

    if target_col not in df.columns:
        raise DataValidationError(f"Target column '{target_col}' not found in dataset.")

    # 3. Handle leakage column
    working_df = df.copy()
    if leakage_col in working_df.columns:
        working_df = working_df.drop(columns=[leakage_col])
        report["dropped_leakage_col"] = True
        logger.info(f"Dropped leakage column '{leakage_col}' from feature set.")

    # 4. Duplicate rows
    duplicates = working_df.duplicated().sum()
    report["duplicate_rows"] = int(duplicates)
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate rows. Dropping duplicates.")
        working_df = working_df.drop_duplicates().reset_index(drop=True)

    # 5. Missing values
    missing = working_df[required_features].isnull().sum().to_dict()
    report["missing_values"] = {k: int(v) for k, v in missing.items() if v > 0}
    total_missing = sum(missing.values())
    if total_missing > 0:
        logger.warning(f"Detected {total_missing} missing values across features.")

    # 6. Infinite values check
    numeric_df = working_df[required_features].select_dtypes(include=[np.number])
    inf_counts = np.isinf(numeric_df).sum().to_dict()
    report["infinite_values"] = {k: int(v) for k, v in inf_counts.items() if v > 0}
    if sum(inf_counts.values()) > 0:
        raise DataValidationError(f"Dataset contains infinite values: {report['infinite_values']}")

    # 7. Numeric range check
    ranges = {}
    for col in required_features:
        col_min = float(working_df[col].min())
        col_max = float(working_df[col].max())
        col_mean = float(working_df[col].mean())
        col_std = float(working_df[col].std())
        ranges[col] = {"min": col_min, "max": col_max, "mean": col_mean, "std": col_std}
    report["feature_ranges"] = ranges

    # 8. Target validation & normalization
    target_series = working_df[target_col].copy()
    if target_series.isnull().any():
        raise DataValidationError("Target column contains missing values.")

    # Normalize target to 0 (Stable) and 1 (Unstable)
    if pd.api.types.is_numeric_dtype(target_series):
        unique_targets = sorted(target_series.unique().tolist())
        if unique_targets not in [[0, 1], [0], [1]]:
            raise DataValidationError(f"Invalid numeric target classes: {unique_targets}. Expected [0, 1].")
        y = target_series.astype(int)
    else:
        # String target (e.g. 'stable', 'unstable')
        target_clean = target_series.astype(str).str.strip().str.lower()
        unrecognized = set(target_clean.unique()) - set(TARGET_MAPPING.keys())
        if unrecognized:
            raise DataValidationError(f"Unrecognized target values found: {unrecognized}")
        y = target_clean.map(TARGET_MAPPING).astype(int)

    # Calculate dynamic target summary
    class_counts = y.value_counts()
    total_samples = len(y)
    target_summary = []
    for cls_val in [0, 1]:
        cnt = int(class_counts.get(cls_val, 0))
        pct = float((cnt / total_samples) * 100.0) if total_samples > 0 else 0.0
        target_summary.append({
            "Class": cls_val,
            "Label": "Stable" if cls_val == 0 else "Unstable",
            "Count": cnt,
            "Percentage": round(pct, 2),
        })
    report["target_summary"] = target_summary

    X = working_df[required_features].copy()
    report["validated_shape"] = X.shape

    return X, y, report

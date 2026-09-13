"""Tests for domain feature engineering transformer."""

import numpy as np
import pandas as pd
import pytest

from config.config import INPUT_FEATURES
from src.feature_engineering import GridFeatureEngineer


def test_feature_engineering_dimensions():
    """Verify that engineered features are appended to input feature matrix."""
    fe = GridFeatureEngineer(include_aggregates=True, include_interactions=True)

    # 12 input features
    df = pd.DataFrame(np.random.rand(15, 12), columns=INPUT_FEATURES)
    fe.fit(df)
    transformed = fe.transform(df)

    # Base (12) + Aggregates (9) + Interactions (4) = 25 features
    assert transformed.shape[1] == 25
    assert not transformed.isnull().values.any()
    assert "total_tau" in transformed.columns
    assert "total_power" in transformed.columns
    assert "p1_g1" in transformed.columns


def test_feature_engineering_mathematical_definitions():
    """Verify exact mathematical formulas for engineered terms."""
    df = pd.DataFrame({
        "tau1": [1.0], "tau2": [2.0], "tau3": [3.0], "tau4": [4.0],
        "p1": [3.0], "p2": [-1.0], "p3": [-1.0], "p4": [-1.0],
        "g1": [0.5], "g2": [0.4], "g3": [0.3], "g4": [0.2],
    })

    fe = GridFeatureEngineer()
    out = fe.fit_transform(df)

    # Check total tau
    assert pytest.approx(out["total_tau"].iloc[0]) == 10.0
    # Check mean tau
    assert pytest.approx(out["mean_tau"].iloc[0]) == 2.5
    # Check net power balance (3 - 1 - 1 - 1 = 0)
    assert pytest.approx(out["total_power"].iloc[0]) == 0.0
    assert pytest.approx(out["power_imbalance"].iloc[0]) == 0.0
    # Check p1*g1 = 3.0 * 0.5 = 1.5
    assert pytest.approx(out["p1_g1"].iloc[0]) == 1.5

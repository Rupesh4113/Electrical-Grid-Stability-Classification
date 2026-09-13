"""Tests for data preprocessing and zero data leakage isolation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from config.config import INPUT_FEATURES
from src.preprocessing import build_pipeline


def test_preprocessing_pipeline_handles_missing_values():
    """Verify median imputer handles missing numeric values seamlessly."""
    pipeline = build_pipeline(classifier=DummyClassifier(strategy="most_frequent"), impute_missing=True)

    X = pd.DataFrame(np.ones((20, len(INPUT_FEATURES))), columns=INPUT_FEATURES)
    X.iloc[0, 0] = np.nan  # Inject missing value
    y = pd.Series([0] * 10 + [1] * 10)

    # Should not raise ValueError about NaN during fit and predict
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == len(X)


def test_zero_leakage_scaling_isolation():
    """Verify scaler parameters are strictly learned from training fold."""
    pipeline = build_pipeline(classifier=DummyClassifier(), scale_features=True)

    X_train = pd.DataFrame(np.full((10, len(INPUT_FEATURES)), 5.0), columns=INPUT_FEATURES)
    y_train = pd.Series([0] * 5 + [1] * 5)
    X_test = pd.DataFrame(np.full((5, len(INPUT_FEATURES)), 10.0), columns=INPUT_FEATURES)

    pipeline.fit(X_train, y_train)
    scaler = pipeline.named_steps["scaler"]

    # Mean learned by scaler should reflect X_train (5.0), unaffected by X_test (10.0)
    # The engineered features will also have their means based purely on X_train
    assert pytest.approx(scaler.mean_[0]) == 5.0

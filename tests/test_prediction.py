"""Tests for inference engine input validation and prediction formatting."""

import numpy as np
import pytest

from config.config import INPUT_FEATURES
from src.prediction import PredictionError, validate_input_features


@pytest.fixture
def valid_input():
    return {
        "tau1": 2.959,
        "tau2": 3.080,
        "tau3": 8.381,
        "tau4": 9.781,
        "p1": 3.763,
        "p2": -1.527,
        "p3": -1.390,
        "p4": -0.845,
        "g1": 0.562,
        "g2": 0.413,
        "g3": 0.778,
        "g4": 0.958,
    }


def test_validate_input_features_success(valid_input):
    """Verify standard valid dictionary yields 1-row DataFrame."""
    df = validate_input_features(valid_input)
    assert len(df) == 1
    assert list(df.columns) == INPUT_FEATURES


def test_validate_input_features_missing_key(valid_input):
    """Verify error on missing key."""
    del valid_input["tau1"]
    with pytest.raises(PredictionError, match="Missing required input features"):
        validate_input_features(valid_input)


def test_validate_input_features_unexpected_key(valid_input):
    """Verify error on extra key."""
    valid_input["extra_param"] = 123.0
    with pytest.raises(PredictionError, match="Unexpected features provided"):
        validate_input_features(valid_input)


def test_validate_input_features_nan_or_inf(valid_input):
    """Verify error on NaN or Inf values."""
    valid_input["tau2"] = float("nan")
    with pytest.raises(PredictionError, match="cannot be NaN"):
        validate_input_features(valid_input)

    valid_input["tau2"] = float("inf")
    with pytest.raises(PredictionError, match="cannot be infinite"):
        validate_input_features(valid_input)


def test_validate_input_features_non_numeric(valid_input):
    """Verify error on non-numeric strings."""
    valid_input["tau3"] = "invalid_string"
    with pytest.raises(PredictionError, match="must be numeric"):
        validate_input_features(valid_input)

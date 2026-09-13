"""Tests for model architectures, training, and prediction interfaces."""

import numpy as np
import pandas as pd
import pytest

from config.config import INPUT_FEATURES
from src.models import get_candidate_models


@pytest.fixture
def sample_dataset():
    """Create a minimal synthetic dataset matching the schema."""
    np.random.seed(42)
    N = 40
    X = pd.DataFrame(np.random.uniform(0.5, 5.0, size=(N, 12)), columns=INPUT_FEATURES)
    # Ensure p features reflect generator/consumer signs
    X["p1"] = np.random.uniform(2.0, 5.0, size=N)
    X["p2"] = -np.random.uniform(0.5, 2.0, size=N)
    X["p3"] = -np.random.uniform(0.5, 2.0, size=N)
    X["p4"] = -(X["p1"] + X["p2"] + X["p3"])
    y = pd.Series(np.random.choice([0, 1], size=N))
    return X, y


def test_all_five_candidate_models_exist():
    """Verify that all 5 required model architectures are registered."""
    models = get_candidate_models()
    expected_models = {"Linear SVM", "Polynomial SVM", "RBF SVM", "KNN", "Decision Tree"}
    assert set(models.keys()) == expected_models


def test_models_fit_and_predict_valid_classes(sample_dataset):
    """Verify each candidate pipeline fits and outputs valid binary classes."""
    X, y = sample_dataset
    models = get_candidate_models()

    for name, (pipeline, _) in models.items():
        # Fit on sample data
        pipeline.fit(X, y)
        preds = pipeline.predict(X)

        assert preds.shape == (len(X),)
        assert set(preds).issubset({0, 1})

        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(X)
            assert probs.shape == (len(X), 2)
            assert np.allclose(probs.sum(axis=1), 1.0)

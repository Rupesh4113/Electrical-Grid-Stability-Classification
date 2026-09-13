"""Inference and real-time prediction engine module."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd

from config.config import (
    BEST_MODEL_PATH,
    INPUT_FEATURES,
    INVERSE_TARGET_MAPPING,
    MODEL_METADATA_PATH,
)

logger = logging.getLogger(__name__)

# Cache for loaded model and metadata in memory
_CACHED_MODEL = None
_CACHED_METADATA = None


class PredictionError(ValueError):
    """Raised when inference input data is malformed or invalid."""
    pass


def load_model(
    model_path: Optional[Union[str, Path]] = None,
    force_reload: bool = False,
):
    """Load serialized model pipeline with in-memory caching."""
    global _CACHED_MODEL
    path = Path(model_path) if model_path else BEST_MODEL_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Please train the model first by running `python train.py`."
        )

    if _CACHED_MODEL is None or force_reload:
        logger.info(f"Loading model pipeline from {path}")
        _CACHED_MODEL = joblib.load(path)

    return _CACHED_MODEL


def load_metadata(
    metadata_path: Optional[Union[str, Path]] = None,
    force_reload: bool = False,
) -> Dict[str, Any]:
    """Load model metadata JSON with in-memory caching."""
    global _CACHED_METADATA
    path = Path(metadata_path) if metadata_path else MODEL_METADATA_PATH

    if not path.exists():
        logger.warning(f"Metadata file not found at {path}.")
        return {}

    if _CACHED_METADATA is None or force_reload:
        with open(path, "r") as f:
            _CACHED_METADATA = json.load(f)

    return _CACHED_METADATA


def validate_input_features(
    input_data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Validate input feature dictionary for schema, types, missingness, and ranges.

    Parameters
    ----------
    input_data : Dict[str, Any]
        Dictionary of input features.
    metadata : Dict[str, Any], optional
        Model metadata containing expected feature ranges.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame ready for model input.
    """
    if not isinstance(input_data, dict):
        raise PredictionError(f"Expected input data as dictionary, got {type(input_data).__name__}.")

    # 1. Missing features check
    missing = [feat for feat in INPUT_FEATURES if feat not in input_data]
    if missing:
        raise PredictionError(f"Missing required input features: {missing}")

    # 2. Unexpected features check
    unexpected = [k for k in input_data.keys() if k not in INPUT_FEATURES]
    if unexpected:
        raise PredictionError(f"Unexpected features provided: {unexpected}. Expected only: {INPUT_FEATURES}")

    # 3. Numeric conversion and NaN/Inf checks
    cleaned_values = {}
    for feat in INPUT_FEATURES:
        val = input_data[feat]
        try:
            num_val = float(val)
        except (ValueError, TypeError):
            raise PredictionError(f"Feature '{feat}' must be numeric, got: {val}")

        if np.isnan(num_val):
            raise PredictionError(f"Feature '{feat}' cannot be NaN.")
        if np.isinf(num_val):
            raise PredictionError(f"Feature '{feat}' cannot be infinite.")

        cleaned_values[feat] = num_val

    # 4. Optional feature range validation based on dataset distributions
    ranges = (metadata or {}).get("feature_ranges", {})
    if ranges:
        for feat, num_val in cleaned_values.items():
            r = ranges.get(feat)
            if r:
                # Allow a reasonable buffer beyond historical min/max (e.g. 5x std)
                std = r.get("std", 1.0)
                min_bound = r.get("min", -1e6) - 3 * std
                max_bound = r.get("max", 1e6) + 3 * std
                if not (min_bound <= num_val <= max_bound):
                    raise PredictionError(
                        f"Feature '{feat}' value {num_val} is outside physically sensible range "
                        f"[{round(min_bound, 2)}, {round(max_bound, 2)}]."
                    )

    return pd.DataFrame([cleaned_values], columns=INPUT_FEATURES)


def predict_stability(
    input_data: Dict[str, Any],
    model_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Predict electrical grid stability (0 = Stable, 1 = Unstable) for a single parameter set.

    Parameters
    ----------
    input_data : Dict[str, Any]
        Dictionary with keys: tau1..tau4, p1..p4, g1..g4.
    model_path : Path, optional
        Path to serialized model pipeline.

    Returns
    -------
    Dict[str, Any]
        Prediction result containing class (0 or 1), label, probabilities, and model info.
    """
    model = load_model(model_path)
    metadata = load_metadata()

    # Validate input
    df_input = validate_input_features(input_data, metadata=metadata)

    # Inference
    pred = int(model.predict(df_input)[0])
    label = INVERSE_TARGET_MAPPING.get(pred, "Unknown")

    # Probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df_input)[0]
        # Class 0 = Stable, Class 1 = Unstable
        prob_stable = float(probs[0])
        prob_unstable = float(probs[1])
    else:
        prob_unstable = float(pred)
        prob_stable = 1.0 - prob_unstable

    model_name = metadata.get("model_name", type(model.named_steps.get("classifier", model)).__name__)

    return {
        "prediction": pred,
        "label": label,
        "probability_stable": round(prob_stable, 4),
        "probability_unstable": round(prob_unstable, 4),
        "model": model_name,
    }

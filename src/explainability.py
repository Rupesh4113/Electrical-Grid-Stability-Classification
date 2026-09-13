"""Explainability and feature importance analysis module."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from config.config import FEATURE_IMPORTANCE_PATH, RANDOM_STATE

logger = logging.getLogger(__name__)

SCIENTIFIC_DISCLAIMER = (
    "Feature importance reflects model behavior and does not establish physical causality."
)


def compute_permutation_importance(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_repeats: int = 10,
    random_state: int = RANDOM_STATE,
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Calculate permutation feature importance on holdout or validation data.

    This works model-agnostically across RBF SVM, Polynomial SVM, Linear SVM, KNN, and Decision Trees,
    measuring the decrease in F1/accuracy when each input feature is shuffled.

    Parameters
    ----------
    pipeline : Pipeline
        Trained model pipeline.
    X_val : pd.DataFrame
        Validation or holdout features.
    y_val : pd.Series
        Validation or holdout true labels.
    n_repeats : int, default 10
        Number of permutation shuffles.
    random_state : int, default RANDOM_STATE
        Random state for shuffling.
    save_path : Path, optional
        Destination CSV path. Defaults to FEATURE_IMPORTANCE_PATH.

    Returns
    -------
    pd.DataFrame
        Ranked feature importance DataFrame with mean and std.
    """
    logger.info("Computing permutation feature importance...")
    perm = permutation_importance(
        pipeline,
        X_val,
        y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="f1",
        n_jobs=-1,
    )

    feature_names = list(X_val.columns)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values(by="importance_mean", ascending=False).reset_index(drop=True)

    dest = save_path or FEATURE_IMPORTANCE_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(dest, index=False)
    logger.info(f"Saved feature importance to {dest}")

    return importance_df


def extract_tree_importances(pipeline: Pipeline, feature_names: list) -> Optional[pd.DataFrame]:
    """Extract native MDI feature importances if the final estimator is a Decision Tree."""
    classifier = pipeline.named_steps.get("classifier")
    if isinstance(classifier, DecisionTreeClassifier):
        if hasattr(classifier, "feature_importances_"):
            # If feature engineer step is present, get engineered names
            if "feature_engineer" in pipeline.named_steps:
                fe = pipeline.named_steps["feature_engineer"]
                names = fe.get_feature_names_out()
            else:
                names = feature_names

            return pd.DataFrame({
                "feature": names,
                "importance": classifier.feature_importances_,
            }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    return None

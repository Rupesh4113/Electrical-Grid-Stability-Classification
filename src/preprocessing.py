"""Data preprocessing and scikit-learn pipeline construction module."""

from typing import Any, List, Optional

from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import GridFeatureEngineer


def build_pipeline(
    classifier: BaseEstimator,
    scale_features: bool = True,
    include_feature_engineering: bool = True,
    impute_missing: bool = True,
) -> Pipeline:
    """Build a complete, zero-leakage scikit-learn Pipeline.

    Architecture:
        [SimpleImputer (median)] -> [GridFeatureEngineer] -> [StandardScaler (optional)] -> [Classifier]

    Parameters
    ----------
    classifier : BaseEstimator
        Scikit-learn classification estimator.
    scale_features : bool, default True
        Whether to standardize features with StandardScaler (essential for distance/margin
        models like SVM and KNN; can be omitted for tree models).
    include_feature_engineering : bool, default True
        Whether to apply domain-specific feature engineering.
    impute_missing : bool, default True
        Whether to include median imputation step for numeric safety.

    Returns
    -------
    Pipeline
        Composed scikit-learn Pipeline instance.
    """
    steps = []

    if impute_missing:
        steps.append(("imputer", SimpleImputer(strategy="median")))

    if include_feature_engineering:
        steps.append(("feature_engineer", GridFeatureEngineer()))

    if scale_features:
        steps.append(("scaler", StandardScaler()))

    steps.append(("classifier", classifier))

    return Pipeline(steps=steps)

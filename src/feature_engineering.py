"""Feature engineering transformer for electrical grid stability analysis."""

from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from config.config import ELASTICITY_FEATURES, INPUT_FEATURES, POWER_FEATURES, TAU_FEATURES


class GridFeatureEngineer(BaseEstimator, TransformerMixin):
    """Domain-informed feature engineering transformer for 4-node grid stability.

    Derives physically and mathematically meaningful variables:
    1. Reaction-time aggregate parameters:
       - total_tau = tau1 + tau2 + tau3 + tau4
       - mean_tau = mean(tau1..tau4)
       - std_tau = standard deviation(tau1..tau4)
    2. Power parameters:
       - total_power = p1 + p2 + p3 + p4 (net systemic power balance)
       - power_imbalance = abs(p1 + p2 + p3 + p4)
       - consumer_power = abs(p2 + p3 + p4)
    3. Elasticity aggregate parameters:
       - total_g = g1 + g2 + g3 + g4
       - mean_g = mean(g1..g4)
       - std_g = standard deviation(g1..g4)
    4. Power-elasticity cross-coupling interactions:
       - p1_g1 = p1 * g1
       - p2_g2 = p2 * g2
       - p3_g3 = p3 * g3
       - p4_g4 = p4 * g4
    """

    def __init__(
        self,
        include_aggregates: bool = True,
        include_interactions: bool = True,
        feature_names: Optional[List[str]] = None,
    ):
        self.include_aggregates = include_aggregates
        self.include_interactions = include_interactions
        self.feature_names = feature_names or INPUT_FEATURES
        self.engineered_feature_names_: List[str] = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Any] = None):
        """Fit transformer by determining output column schema."""
        self._validate_input_columns(X)
        self._compute_feature_names()
        return self

    def _validate_input_columns(self, X: Union[pd.DataFrame, np.ndarray]):
        if isinstance(X, pd.DataFrame):
            self.input_feature_names_ = list(X.columns)
        else:
            self.input_feature_names_ = list(self.feature_names)

    def _compute_feature_names(self):
        names = list(self.input_feature_names_)

        if self.include_aggregates:
            names.extend([
                "total_tau",
                "mean_tau",
                "std_tau",
                "total_power",
                "power_imbalance",
                "consumer_power",
                "total_g",
                "mean_g",
                "std_g",
            ])

        if self.include_interactions:
            names.extend([
                "p1_g1",
                "p2_g2",
                "p3_g3",
                "p4_g4",
            ])

        self.engineered_feature_names_ = names

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Transform input features by deriving engineered grid metrics."""
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=self.input_feature_names_)

        # Extract base arrays
        tau_cols = [c for c in TAU_FEATURES if c in df.columns]
        p_cols = [c for c in POWER_FEATURES if c in df.columns]
        g_cols = [c for c in ELASTICITY_FEATURES if c in df.columns]

        if self.include_aggregates:
            if len(tau_cols) == 4:
                df["total_tau"] = df[tau_cols].sum(axis=1)
                df["mean_tau"] = df[tau_cols].mean(axis=1)
                df["std_tau"] = df[tau_cols].std(axis=1, ddof=0)

            if len(p_cols) == 4:
                df["total_power"] = df[p_cols].sum(axis=1)
                df["power_imbalance"] = df["total_power"].abs()
                df["consumer_power"] = df[["p2", "p3", "p4"]].sum(axis=1).abs()

            if len(g_cols) == 4:
                df["total_g"] = df[g_cols].sum(axis=1)
                df["mean_g"] = df[g_cols].mean(axis=1)
                df["std_g"] = df[g_cols].std(axis=1, ddof=0)

        if self.include_interactions:
            for i in range(1, 5):
                p_col = f"p{i}"
                g_col = f"g{i}"
                if p_col in df.columns and g_col in df.columns:
                    df[f"p{i}_g{i}"] = df[p_col] * df[g_col]

        return df

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Get output feature names."""
        return np.array(self.engineered_feature_names_)

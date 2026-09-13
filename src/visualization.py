"""Visualization module providing interactive Plotly and static charts."""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config.config import (
    ELASTICITY_FEATURES,
    INPUT_FEATURES,
    INVERSE_TARGET_MAPPING,
    POWER_FEATURES,
    TAU_FEATURES,
)

PCA_DISCLAIMER = (
    "PCA projection is for visualization and does not represent the complete 12-dimensional decision boundary."
)


def plot_target_distribution(y: pd.Series) -> go.Figure:
    """Create interactive Plotly bar chart for target class balance."""
    counts = y.value_counts().reset_index()
    counts.columns = ["Target", "Count"]
    counts["Label"] = counts["Target"].map(INVERSE_TARGET_MAPPING)
    counts["Percentage"] = (counts["Count"] / len(y) * 100).round(2)

    fig = px.bar(
        counts,
        x="Label",
        y="Count",
        color="Label",
        color_discrete_map={"Stable": "#2ca02c", "Unstable": "#d62728"},
        text="Count",
        title="Electrical Grid Stability Class Distribution",
    )
    fig.update_traces(
        texttemplate="%{text} (%{customdata:.1f}%)",
        customdata=counts["Percentage"],
        textposition="outside",
    )
    fig.update_layout(
        xaxis_title="Stability State",
        yaxis_title="Observation Count",
        showlegend=False,
        template="plotly_white",
    )
    return fig


def plot_feature_distribution(df: pd.DataFrame, feature_name: str) -> go.Figure:
    """Plot distribution of a specific feature grouped by stability state."""
    df_plot = df.copy()
    if "stabf" in df_plot.columns:
        label_col = "stabf"
    elif "target" in df_plot.columns:
        df_plot["label"] = df_plot["target"].map(INVERSE_TARGET_MAPPING)
        label_col = "label"
    else:
        label_col = None

    fig = px.histogram(
        df_plot,
        x=feature_name,
        color=label_col,
        barmode="overlay",
        marginal="box",
        opacity=0.6,
        color_discrete_map={"Stable": "#2ca02c", "Unstable": "#d62728", "stable": "#2ca02c", "unstable": "#d62728"},
        title=f"Distribution of {feature_name} by Grid Stability",
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_correlation_matrix(df: pd.DataFrame, features: Optional[List[str]] = None) -> go.Figure:
    """Plot interactive correlation heatmap for electrical and stability features."""
    cols = features or INPUT_FEATURES
    corr = df[cols].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Feature Correlation Matrix",
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_feature_boxplots(df: pd.DataFrame, feature_group: str = "tau") -> go.Figure:
    """Plot boxplots for feature groups (tau, power, or elasticity) partitioned by stability."""
    group_map = {
        "tau": TAU_FEATURES,
        "power": POWER_FEATURES,
        "elasticity": ELASTICITY_FEATURES,
    }
    cols = group_map.get(feature_group.lower(), TAU_FEATURES)
    target_col = "stabf" if "stabf" in df.columns else "target"

    df_melted = pd.melt(
        df,
        id_vars=[target_col],
        value_vars=cols,
        var_name="Feature",
        value_name="Value",
    )
    df_melted["Label"] = df_melted[target_col].map(
        lambda x: INVERSE_TARGET_MAPPING.get(x, str(x).capitalize())
    )

    fig = px.box(
        df_melted,
        x="Feature",
        y="Value",
        color="Label",
        color_discrete_map={"Stable": "#2ca02c", "Unstable": "#d62728"},
        title=f"Boxplot Comparison of {feature_group.capitalize()} Parameters by Stability",
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_kernel_comparison(kernel_df: pd.DataFrame) -> go.Figure:
    """Create multi-metric comparative bar chart for SVM kernels."""
    fig = go.Figure()

    metrics = ["cv_accuracy", "cv_f1", "test_accuracy", "test_f1"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for metric, color in zip(metrics, colors):
        if metric in kernel_df.columns:
            fig.add_trace(
                go.Bar(
                    name=metric.replace("_", " ").title(),
                    x=kernel_df["kernel"],
                    y=kernel_df[metric],
                    marker_color=color,
                )
            )

    fig.update_layout(
        barmode="group",
        title="SVM Kernel Comparison Across Evaluation Metrics",
        xaxis_title="Kernel",
        yaxis_title="Score",
        yaxis=dict(range=[0.7, 1.0]),
        template="plotly_white",
    )
    return fig


def compute_pca_projection(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, PCA]:
    """Standardize features and project onto top 2 principal components for visualization."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
    pca_df["Target"] = y.values
    pca_df["Label"] = pca_df["Target"].map(INVERSE_TARGET_MAPPING)

    return pca_df, pca


def plot_pca_scatter(pca_df: pd.DataFrame, pca_model: PCA) -> go.Figure:
    """Scatter plot of 2D PCA projection of the grid parameter space."""
    var1 = pca_model.explained_variance_ratio_[0] * 100
    var2 = pca_model.explained_variance_ratio_[1] * 100

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Label",
        color_discrete_map={"Stable": "#2ca02c", "Unstable": "#d62728"},
        opacity=0.6,
        title=f"2D PCA Projection of Grid Parameters (Explained Variance: {var1:.1f}% + {var2:.1f}%)",
    )
    fig.add_annotation(
        text=PCA_DISCLAIMER,
        xref="paper",
        yref="paper",
        x=0,
        y=-0.2,
        showarrow=False,
        font=dict(size=10, color="gray"),
    )
    fig.update_layout(template="plotly_white")
    return fig

"""Model evaluation, cross-validation, and metrics generation module."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from config.config import (
    CONFUSION_MATRIX_PATH,
    CV_SPLITS,
    METRICS_PATH,
    MODEL_COMPARISON_PATH,
    PRIMARY_METRIC,
    RANDOM_STATE,
)

logger = logging.getLogger(__name__)


def evaluate_cross_validation(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = CV_SPLITS,
    random_state: int = RANDOM_STATE,
) -> Dict[str, float]:
    """Execute stratified 5-fold cross-validation and compute mean and std across metrics.

    Parameters
    ----------
    pipeline : Pipeline
        Candidate scikit-learn pipeline.
    X : pd.DataFrame
        Training features.
    y : pd.Series
        Training binary targets (0/1).
    cv_splits : int, default CV_SPLITS
        Number of stratified splits.
    random_state : int, default RANDOM_STATE
        Random state for fold shuffling.

    Returns
    -------
    Dict[str, float]
        Cross-validation metrics including mean and standard deviation.
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "balanced_accuracy": "balanced_accuracy",
        "roc_auc": "roc_auc",
    }

    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    results = {
        "mean_accuracy": float(np.mean(cv_results["test_accuracy"])),
        "std_accuracy": float(np.std(cv_results["test_accuracy"])),
        "mean_f1": float(np.mean(cv_results["test_f1"])),
        "std_f1": float(np.std(cv_results["test_f1"])),
        "mean_precision": float(np.mean(cv_results["test_precision"])),
        "std_precision": float(np.std(cv_results["test_precision"])),
        "mean_recall": float(np.mean(cv_results["test_recall"])),
        "std_recall": float(np.std(cv_results["test_recall"])),
        "mean_balanced_accuracy": float(np.mean(cv_results["test_balanced_accuracy"])),
        "std_balanced_accuracy": float(np.std(cv_results["test_balanced_accuracy"])),
        "mean_roc_auc": float(np.mean(cv_results["test_roc_auc"])),
        "std_roc_auc": float(np.std(cv_results["test_roc_auc"])),
    }
    return results


def evaluate_holdout(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """Perform a single, rigorous evaluation of the finalized model on the isolated test set.

    Parameters
    ----------
    pipeline : Pipeline
        Trained model pipeline.
    X_test : pd.DataFrame
        Holdout test features.
    y_test : pd.Series
        Holdout test labels.
    save_artifacts : bool, default True
        Whether to save confusion matrix image and metrics JSON.

    Returns
    -------
    Dict[str, Any]
        Holdout evaluation metrics and diagnostic breakdown.
    """
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=["Stable (0)", "Unstable (1)"],
        output_dict=True,
    )

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "balanced_accuracy": float(bal_acc),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "matrix": cm.tolist(),
        },
        "classification_report": report_dict,
        "sample_size": len(y_test),
    }

    if save_artifacts:
        save_evaluation_artifacts(cm, metrics)

    return metrics


def save_evaluation_artifacts(
    cm: np.ndarray,
    metrics: Dict[str, Any],
    cm_path: Optional[Path] = None,
    metrics_path: Optional[Path] = None,
):
    """Save confusion matrix visualization and holdout metrics to artifacts directory."""
    cm_file = cm_path or CONFUSION_MATRIX_PATH
    metrics_file = metrics_path or METRICS_PATH

    cm_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved holdout metrics to {metrics_file}")

    # Generate and save Confusion Matrix plot
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Stable (0)", "Unstable (1)"],
        yticklabels=["Stable (0)", "Unstable (1)"],
        cbar=True,
    )
    plt.title("Holdout Test Confusion Matrix\nElectrical Grid Stability", fontsize=13, pad=15)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(cm_file, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {cm_file}")

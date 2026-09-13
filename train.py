"""Master training and benchmarking script for Electrical Grid Stability Classification."""

import hashlib
import json
import logging
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from benchmark import run_latency_benchmark
from config.config import (
    BEST_MODEL_PATH,
    CONFUSION_MATRIX_PATH,
    CV_SPLITS,
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_METADATA_PATH,
    PRIMARY_METRIC,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data_loader import load_raw_data
from src.evaluation import evaluate_cross_validation, evaluate_holdout
from src.explainability import SCIENTIFIC_DISCLAIMER, compute_permutation_importance
from src.models import get_candidate_models
from src.validation import validate_raw_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def compute_dataset_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of dataset file for provenance tracking."""
    if not filepath.exists():
        return "not_found"
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def run_training_pipeline():
    """Execute end-to-end training, hyperparameter tuning, model comparison, and serialization."""
    logger.info("=" * 60)
    logger.info("STARTING ELECTRICAL GRID STABILITY TRAINING PIPELINE")
    logger.info("=" * 60)

    # 1. Load raw dataset
    logger.info("Step 1: Ingesting dataset...")
    df_raw = load_raw_data()

    # 2. Validate dataset schema and integrity
    logger.info("Step 2: Validating schema, class distribution, and data integrity...")
    X, y, val_report = validate_raw_dataset(df_raw)
    logger.info(f"Validated dataset shape: {X.shape}, Target distribution: {val_report['target_summary']}")

    # 3. Stratified Train/Test Split
    logger.info(f"Step 3: Performing Stratified Train/Test Split ({100 - int(TEST_SIZE*100)}/{int(TEST_SIZE*100)})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    logger.info(f"Training split: {X_train.shape[0]} samples, Holdout test: {X_test.shape[0]} samples.")

    # 4. Stratified 5-Fold Cross Validation & Hyperparameter Optimization for 5 Models
    logger.info(f"Step 4: Hyperparameter Optimization and Stratified {CV_SPLITS}-Fold Cross-Validation...")
    candidate_models = get_candidate_models(random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    model_comparison = []
    trained_pipelines = {}
    best_estimators = {}

    for model_name, (pipeline, param_grid) in candidate_models.items():
        logger.info(f"\n--- Tuning and Evaluating: {model_name} ---")
        t_start = time.perf_counter()

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring=PRIMARY_METRIC,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
        )
        grid_search.fit(X_train, y_train)
        fit_time = time.perf_counter() - t_start

        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        best_estimators[model_name] = best_pipeline

        # Comprehensive CV metrics on the best estimator
        cv_metrics = evaluate_cross_validation(best_pipeline, X_train, y_train, cv_splits=CV_SPLITS)

        logger.info(
            f"{model_name} Best CV F1: {best_score:.4f} "
            f"(Accuracy: {cv_metrics['mean_accuracy']:.4f} ± {cv_metrics['std_accuracy']:.4f}) "
            f"in {fit_time:.2f}s"
        )
        logger.info(f"Best Hyperparameters: {best_params}")

        # Also evaluate on holdout test set for complete comparison table
        test_preds = best_pipeline.predict(X_test)
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        test_acc = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds, zero_division=0)
        test_prec = precision_score(y_test, test_preds, zero_division=0)
        test_rec = recall_score(y_test, test_preds, zero_division=0)

        # Measure quick inference latency on test set
        t_inf_start = time.perf_counter()
        _ = best_pipeline.predict(X_test.iloc[:100])
        inf_latency_per_sample_ms = ((time.perf_counter() - t_inf_start) / 100.0) * 1000.0

        model_comparison.append({
            "model": model_name,
            "cv_f1_mean": round(cv_metrics["mean_f1"], 4),
            "cv_f1_std": round(cv_metrics["std_f1"], 4),
            "cv_accuracy_mean": round(cv_metrics["mean_accuracy"], 4),
            "cv_accuracy_std": round(cv_metrics["std_accuracy"], 4),
            "cv_precision_mean": round(cv_metrics["mean_precision"], 4),
            "cv_recall_mean": round(cv_metrics["mean_recall"], 4),
            "test_accuracy": round(test_acc, 4),
            "test_f1": round(test_f1, 4),
            "test_precision": round(test_prec, 4),
            "test_recall": round(test_rec, 4),
            "training_time_sec": round(fit_time, 2),
            "inference_latency_ms": round(inf_latency_per_sample_ms, 4),
            "best_params": str(best_params),
        })

    # Save model comparison table
    df_comparison = pd.DataFrame(model_comparison).sort_values(by="cv_f1_mean", ascending=False).reset_index(drop=True)
    df_comparison.to_csv(MODEL_COMPARISON_PATH, index=False)
    logger.info(f"\nModel Comparison Summary:\n{df_comparison[['model', 'cv_f1_mean', 'cv_accuracy_mean', 'test_f1', 'test_accuracy']]}")

    # 5. Model Selection (Dynamic selection based on primary metric: CV F1)
    winner_name = df_comparison.iloc[0]["model"]
    winner_pipeline = best_estimators[winner_name]
    logger.info(f"\nWinner Model Selected based on CV F1: {winner_name}")

    # 6. Final Holdout Test Evaluation
    logger.info("Step 5: Evaluating winner on isolated holdout test set (exactly once)...")
    winner_pipeline.fit(X_train, y_train)
    holdout_metrics = evaluate_holdout(winner_pipeline, X_test, y_test, save_artifacts=True)
    logger.info(f"Holdout Test Accuracy: {holdout_metrics['accuracy']:.4f}, F1: {holdout_metrics['f1']:.4f}")

    # 7. Model Explainability
    logger.info("Step 6: Generating permutation feature importance...")
    perm_importance_df = compute_permutation_importance(
        winner_pipeline,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        save_path=FEATURE_IMPORTANCE_PATH,
    )
    logger.info(f"Top 5 most influential features in model:\n{perm_importance_df.head(5)}")

    # 8. Model Serialization
    logger.info("Step 7: Serializing model pipeline and metadata...")
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(winner_pipeline, BEST_MODEL_PATH)
    logger.info(f"Serialized winner model to {BEST_MODEL_PATH}")

    # Create metadata JSON
    from config.config import DATASET_RAW_PATH
    metadata = {
        "model_name": winner_name,
        "model_version": "1.0.0",
        "training_timestamp": datetime.utcnow().isoformat(),
        "dataset_hash": compute_dataset_hash(DATASET_RAW_PATH),
        "primary_selection_metric": PRIMARY_METRIC,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "input_features": list(X.columns),
        "target_mapping": {"0": "Stable", "1": "Unstable"},
        "feature_ranges": val_report["feature_ranges"],
        "best_hyperparameters": eval(df_comparison.loc[df_comparison["model"] == winner_name, "best_params"].values[0]),
        "cv_performance": {
            "f1_mean": float(df_comparison.loc[df_comparison["model"] == winner_name, "cv_f1_mean"].values[0]),
            "f1_std": float(df_comparison.loc[df_comparison["model"] == winner_name, "cv_f1_std"].values[0]),
            "accuracy_mean": float(df_comparison.loc[df_comparison["model"] == winner_name, "cv_accuracy_mean"].values[0]),
            "accuracy_std": float(df_comparison.loc[df_comparison["model"] == winner_name, "cv_accuracy_std"].values[0]),
        },
        "holdout_performance": {
            "accuracy": holdout_metrics["accuracy"],
            "f1": holdout_metrics["f1"],
            "precision": holdout_metrics["precision"],
            "recall": holdout_metrics["recall"],
            "balanced_accuracy": holdout_metrics["balanced_accuracy"],
            "roc_auc": holdout_metrics["roc_auc"],
        },
        "packages": {
            "scikit-learn": __import__("sklearn").__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        "scientific_disclaimer": SCIENTIFIC_DISCLAIMER,
    }

    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Saved metadata to {MODEL_METADATA_PATH}")

    # 9. Run Latency Benchmark
    logger.info("Step 8: Benchmarking inference latency...")
    run_latency_benchmark(model_path=BEST_MODEL_PATH)

    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE COMPLETE SUCCESSFULLY!")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_training_pipeline()

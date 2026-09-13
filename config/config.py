"""Configuration settings for Electrical Grid Stability Classification."""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Ensure directories exist
for p in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, ARTIFACTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET_URLS = [
    "https://archive.ics.uci.edu/static/public/471/electrical+grid+stability+simulated+data.zip",
    "https://raw.githubusercontent.com/Arin-Saha/Electrical-Grid-Stability-Simulated-Data/master/Data_for_UCI_named.csv",
    "https://raw.githubusercontent.com/chakraborty-arnab/Electrical-Grid-Stability-Simulated-Data-Set/master/Data_for_UCI_named.csv",
]
DATASET_CSV_FILENAME = "Data_for_UCI_named.csv"
DATASET_RAW_PATH = DATA_RAW_DIR / DATASET_CSV_FILENAME

# Features & Target definitions
TAU_FEATURES = ["tau1", "tau2", "tau3", "tau4"]
POWER_FEATURES = ["p1", "p2", "p3", "p4"]
ELASTICITY_FEATURES = ["g1", "g2", "g3", "g4"]
INPUT_FEATURES = TAU_FEATURES + POWER_FEATURES + ELASTICITY_FEATURES

TARGET_COL = "stabf"
LEAKAGE_COL = "stab"  # Continuous root determining stabf; must be excluded from inputs

TARGET_MAPPING = {
    "stable": 0,
    "unstable": 1,
}
INVERSE_TARGET_MAPPING = {0: "Stable", 1: "Unstable"}

# Train/Test Split
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Cross Validation
CV_SPLITS = 5
PRIMARY_METRIC = "f1"

# Hyperparameter Search Grids
PARAM_GRIDS = {
    "Linear SVM": {
        "classifier__C": [0.1, 1.0, 10.0],
    },
    "Polynomial SVM": {
        "classifier__C": [0.1, 1.0, 10.0, 100.0],
        "classifier__degree": [2, 3, 4],
        "classifier__gamma": ["scale", 0.01, 0.1],
    },
    "RBF SVM": {
        "classifier__C": [0.1, 1.0, 10.0, 100.0],
        "classifier__gamma": [0.001, 0.01, 0.1, 1.0],
    },
    "KNN": {
        "classifier__n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25],
        "classifier__weights": ["uniform", "distance"],
        "classifier__p": [1, 2],
    },
    "Decision Tree": {
        "classifier__max_depth": [2, 4, 6, 8, 10, 12, 14],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__criterion": ["gini", "entropy"],
    },
}

# Model and Artifact Filepaths
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
MODEL_COMPARISON_PATH = ARTIFACTS_DIR / "model_comparison.csv"
CONFUSION_MATRIX_PATH = ARTIFACTS_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance.csv"
INFERENCE_BENCHMARK_PATH = ARTIFACTS_DIR / "inference_benchmark.json"

# API Configuration
API_HOST = "127.0.0.1"
API_PORT = 8000
API_URL = f"http://{API_HOST}:{API_PORT}"

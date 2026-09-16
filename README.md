# Electrical Grid Stability Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)

A production-quality machine learning system and benchmarking framework for binary classification of decentralized electrical power-grid synchronization stability (`0 = Stable`, `1 = Unstable`).

---

## Overview

Modern smart electrical grids increasingly rely on decentralized topologies where consumers and distributed energy resources (DERs) dynamically adjust consumption and generation in response to real-time price signals. However, dynamic response delays and elasticity variations can trigger synchronization loss and electromechanical oscillations. 

This repository provides an end-to-end, zero-data-leakage machine learning framework evaluating five distinct algorithmic paradigms:
1. **Support Vector Machine — Linear Kernel**
2. **Support Vector Machine — Polynomial Kernel**
3. **Support Vector Machine — Radial Basis Function (RBF) Kernel**
4. **K-Nearest Neighbors (KNN)**
5. **Decision Tree Classifier**

The project includes domain-specific feature engineering, nested cross-validation, permutation explainability, sub-millisecond latency benchmarking, a FastAPI REST service, and an interactive Streamlit visualization dashboard.

---

## Research Problem

In decentralized power systems governed by the swing equation:
$$\frac{2H}{\omega_s} \frac{d^2\delta_i}{dt^2} = P_{m,i} - P_{e,i} - D_i \frac{d\delta_i}{dt}$$
participants adjust demand or generation with reaction delay $\tau_i$ and elasticity $g_i$. When delays exceed stability margins or power imbalances grow, the system loses asymptotic phase synchronization. 

This benchmark evaluates which statistical learning architectures best capture the nonlinear boundary between synchronized equilibrium and dynamical instability without manual feature thresholding.

---

## Dataset

The analysis utilizes the **Electrical Grid Stability Simulated Data Set** (UCI Machine Learning Repository ID: 471), contributed by Vadim Arzamasov, Klemens Böhm, and Patrick Jochem (Karlsruhe Institute of Technology).

- **Total Observations**: 10,000 simulations
- **Topology**: 4-node star architecture (1 central power supplier node, 3 consumer nodes)
- **Input Dimensions**: 12 electrical and dynamic system features:
  - **Reaction-time parameters ($\tau_1 - \tau_4$)**: Time constants (0.5 to 10.0 s) for participants to adjust consumption/generation.
  - **Power parameters ($p_1 - p_4$)**: Net power generated ($p_1 > 0$, 0.5 to 6.0 MW) or consumed ($p_2, p_3, p_4 < 0$, -2.0 to -0.5 MW), subject to balance $\sum_{i=1}^4 p_i = 0$.
  - **Elasticity coefficients ($g_1 - g_4$)**: Price elasticity parameters (0.05 to 1.0) governing participant demand response.

---

## Target Definition

- **Target Column**: `stabf`
- **Normalization Mapping**:
  - `0 = Stable` (Grid returns to synchronous operating point)
  - `1 = Unstable` (Grid diverges or enters non-synchronous oscillations)
- **Zero-Leakage Guarantee**: The continuous root variable `stab` (the maximal real part of characteristic equation roots used in the original simulation to assign `stabf`) is strictly excluded from input feature matrices prior to training.

---

## Data Pipeline

The pipeline enforces strict data leakage prevention:
```
Raw Ingestion (data/raw/Data_for_UCI_named.csv)
      │
      ▼
Integrity Validation (types, bounds, missingness, leakage removal)
      │
      ▼
Stratified Train/Test Split (80% Train / 20% Holdout)
      │
      ▼  [Pipeline Boundary]
      ├─ SimpleImputer (median)
      ├─ GridFeatureEngineer (aggregate & interaction terms)
      ├─ StandardScaler (fitted strictly on training fold)
      └─ Classifier (Hyperparameter Tuning via Stratified 5-Fold CV)
```

All scalers and feature transformers are fitted exclusively inside cross-validation training folds. The holdout test set is evaluated exactly once on the finalized winner pipeline.

---

## Exploratory Data Analysis

EDA is available in [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) and in the Streamlit application:
- **Class Balance**: 6,380 (63.8%) Unstable observations vs 3,620 (36.2%) Stable observations.
- **Reaction Time Trends**: Higher reaction times across $\tau_1 - \tau_4$ show strong positive association with instability.
- **Power Flow Symmetry**: Power flows adhere to net physical balance ($p_1 + p_2 + p_3 + p_4 \approx 0$).
- **Correlation Structure**: Base input features exhibit low linear collinearity, highlighting that stability boundaries are governed by nonlinear multi-variable interactions.

---

## Feature Engineering

Implemented as a reusable scikit-learn transformer (`GridFeatureEngineer`):
- **Total Reaction Time Index**: $\tau_{\text{total}} = \sum_{i=1}^4 \tau_i$
- **Mean & Dispersion of Reaction Times**: $\bar{\tau} = \frac{1}{4} \sum \tau_i$, $\sigma_\tau = \text{std}(\tau_1..\tau_4)$
- **Net Power Imbalance**: $|p_1 + p_2 + p_3 + p_4|$
- **Total Elasticity & Mean Elasticity**: $g_{\text{total}} = \sum g_i$, $\bar{g} = \text{mean}(g_1..g_4)$
- **Coupled Power-Elasticity Cross Terms**: $p_i \cdot g_i$ for $i \in \{1, 2, 3, 4\}$

---

## Machine Learning Models

Five models are implemented and tuned:
1. **Linear SVM**: Baseline margin classifier testing linear separability.
2. **Polynomial SVM**: Tests degree-2 to 4 polynomial boundary surfaces to capture multi-participant power interactions.
3. **Radial Basis Function (RBF) SVM**: Flexible infinite-dimensional kernel mapping for localized nonlinear stability pockets.
4. **K-Nearest Neighbors (KNN)**: Non-parametric distance-based classification testing metric locality.
5. **Decision Tree**: Non-linear axis-aligned partitioning without distance scaling requirements.

---

## Hyperparameter Optimization

Hyperparameters are systematically explored using `GridSearchCV` over **Stratified 5-Fold Cross-Validation**:
- **Primary Optimization Metric**: **F1-Score** (penalizes false alarms and missed instability events equally under asymmetric operational risk).
- **Secondary Metrics**: Accuracy, Precision, Recall, Balanced Accuracy, ROC-AUC.

---

## Validation Strategy

- **Stratified 5-Fold Cross-Validation**: Folds preserve the empirical 63.8% / 36.2% class distribution.
- **Independent Holdout**: 2,000 test observations (20%) held isolated until model selection completes.
- **Reproducibility**: Global seed `random_state=42`.

---

## Model Comparison

Summary of actual cross-validation and holdout test performance (calculated directly from dataset execution):

| Model | CV F1 (Mean ± Std) | CV Accuracy | Holdout F1 | Holdout Accuracy | Holdout ROC-AUC | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RBF SVM** (Winner) | **0.9629 ± 0.0015** | **95.26%** | **0.9686** | **96.00%** | **0.9944** | 702.47s |
| **Polynomial SVM** | 0.9426 ± 0.0049 | 92.64% | 0.9538 | 94.15% | 0.9856 | 2333.26s |
| **KNN** | 0.9069 ± 0.0055 | 87.42% | 0.9097 | 87.75% | 0.9463 | 40.16s |
| **Linear SVM** | 0.8928 ± 0.0050 | 86.27% | 0.9035 | 87.70% | 0.9421 | 176.72s |
| **Decision Tree** | 0.8854 ± 0.0059 | 85.45% | 0.8951 | 86.50% | 0.9063 | 80.60s |

*(Full metrics automatically serialized to `artifacts/model_comparison.csv` and `artifacts/metrics.json`).*

---

## Explainability

Permutation feature importance is computed on the isolated holdout test set using 10 shuffle iterations:
- Measures the drop in test F1 score when feature columns are scrambled.
- **Scientific Caveat**: Feature importance reflects model behavior and does not establish physical causality.

---

## Inference Benchmark

Measured using `time.perf_counter()` on the complete serialized pipeline (including imputer, feature engineering, scaler, and estimator):
- **Single-Sample Latency**: Average, median, P95, and P99 latencies recorded across 500 warmup-buffered iterations.
- **Batch Latency**: Throughput scaled across batch sizes 1 to 1,000.
- **Artifact**: `artifacts/inference_benchmark.json`.

---

## FastAPI REST Service

FastAPI service located at [`api/main.py`](api/main.py).

### Endpoints
- `GET /`: Service metadata and available endpoints.
- `GET /health`: Health status and loaded model verification.
- `GET /model-info`: Hyperparameters, train date, holdout metrics.
- `POST /predict`: Real-time prediction with Pydantic validation.

### Sample Request
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "tau1": 2.959, "tau2": 3.080, "tau3": 8.381, "tau4": 9.781,
       "p1": 3.763, "p2": -1.527, "p3": -1.390, "p4": -0.845,
       "g1": 0.562, "g2": 0.413, "g3": 0.778, "g4": 0.958
     }'
```

### Sample Response
```json
{
  "prediction": 1,
  "label": "Unstable",
  "probability_stable": 0.052,
  "probability_unstable": 0.948,
  "model": "RBF SVM"
}
```

---

## Streamlit Dashboard

Multi-page interactive application in [`app.py`](app.py):
1. **Overview**: Key metrics, architecture, dataset summary.
2. **Stability Prediction**: Interactive sliders, dynamic system balance feedback, gauge visualization. Supports **Local Pipeline** and **FastAPI REST** modes.
3. **Model Comparison**: Full comparative benchmark table, holdout bar charts, confusion matrix.
4. **Kernel Analysis**: Linear vs Polynomial vs RBF SVM comparative analysis with theoretical explanations.
5. **Feature Importance**: Permutation importance bar chart with operational caveats.
6. **Data Exploration**: Univariate histograms, group boxplots, correlation heatmap, and 2D PCA decision space projection.
7. **Inference Benchmark**: Real latency percentiles and throughput scaling charts.
8. **About & Limitations**: Methodological safeguards, citations, scientific language compliance, limitations.

---

## Installation

```bash
git clone https://github.com/Rupesh4113/Electrical-Grid-Stability-Classification.git
cd "Electrical Grid Stability Classification"
pip install -r requirements.txt
```

---

## Training

Execute the end-to-end training and benchmark pipeline:
```bash
python train.py
```
This executes data ingestion, validation, hyperparameter tuning, model selection, holdout evaluation, explainability, latency benchmarking, and serializes all artifacts.

---

## Testing

Run the automated pytest test suite:
```bash
python -m pytest -v
```

---

## Running Streamlit

Launch the dashboard:
```bash
streamlit run app.py
```

---

## Running API

Launch the production REST service:
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger documentation is available at `http://127.0.0.1:8000/docs`.

---

## Project Architecture

```
Electrical Grid Stability Classification/
├── app.py                     # Interactive Streamlit dashboard
├── train.py                   # Master end-to-end training pipeline
├── predict.py                 # CLI inference utility
├── benchmark.py               # Precision latency benchmarking
├── requirements.txt           # Production dependencies
├── README.md                  # Comprehensive documentation
├── LICENSE                    # Apache 2.0 license
│
├── config/
│   └── config.py              # Centralized configuration & parameters
│
├── data/
│   ├── raw/                   # UCI dataset CSV
│   └── processed/             # Cleaned & partitioned data
│
├── models/
│   ├── best_model.joblib      # Serialized scikit-learn winner pipeline
│   └── model_metadata.json    # Provenance, hyperparams, & metrics
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Resilient download & data ingestion
│   ├── validation.py          # Data schema & integrity checks
│   ├── feature_engineering.py # Domain GridFeatureEngineer transformer
│   ├── preprocessing.py       # Zero-leakage scikit-learn pipelines
│   ├── models.py              # Model candidates & hyperparameter grids
│   ├── evaluation.py          # Stratified CV & holdout evaluation
│   ├── explainability.py      # Permutation importance
│   ├── prediction.py          # Real-time inference engine
│   └── visualization.py       # Plotly & Matplotlib visualization
│
├── api/
│   ├── __init__.py
│   └── main.py                # FastAPI REST API service
│
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   └── 02_model_analysis.ipynb# Model benchmarking & kernel diagnostics
│
├── tests/
│   ├── test_data.py           # Ingestion & validation tests
│   ├── test_preprocessing.py  # Leakage isolation & preprocessing tests
│   ├── test_features.py       # Feature engineering unit tests
│   ├── test_models.py         # Architecture training & prediction tests
│   ├── test_prediction.py     # Inference engine & schema tests
│   └── test_api.py            # FastAPI endpoint tests
│
└── artifacts/
    ├── metrics.json           # Final holdout test metrics
    ├── model_comparison.csv   # Complete comparative leaderboard
    ├── confusion_matrix.png   # Normalized holdout confusion matrix
    ├── feature_importance.csv # Permutation importance rankings
    └── inference_benchmark.json # Latency percentiles & hardware info
```

---

## Scientific Limitations & Engineering Disclaimers

> ⚠️ **CRITICAL ENGINEERING DISCLAIMER**  
> **This model should not be used for operational grid control without extensive domain validation, safety analysis, and regulatory/engineering approval.**

1. **Simulation-Based Nature**: The dataset reflects synthetic simulations derived from the differential swing equation on a 4-node star topology. Physical power systems encompass thousands of buses, asymmetric three-phase imbalances, harmonic distortions, and stochastic weather/inverter dynamics.
2. **Domain Shift Risk**: Models trained purely on numerical simulation are vulnerable to distribution shift when deployed against raw phasor measurement unit (PMU) telemetry.
3. **Absence of Real-Time Telemetry Validation**: In situ field telemetry testing under hardware-in-the-loop (HIL) conditions has not been conducted.
4. **Computational Complexity**: While RBF SVM demonstrates superior separation on the evaluated 10,000 samples, support vector optimization exhibits super-linear $\mathcal{O}(N^2 \text{ to } N^3)$ computational scaling, necessitating alternative surrogate models or linear approximations for large interconnected grids.
5. **Correlation vs. Causality**: The model and sensitivity analysis identified parameter regions associated with higher predicted instability within the simulation dataset. Feature importance measures statistical sensitivity in the model and does not establish physical causality.

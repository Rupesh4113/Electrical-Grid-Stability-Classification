"""Streamlit Web Application for Electrical Grid Stability Classification."""

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from config.config import (
    API_URL,
    ARTIFACTS_DIR,
    BEST_MODEL_PATH,
    CONFUSION_MATRIX_PATH,
    DATASET_RAW_PATH,
    FEATURE_IMPORTANCE_PATH,
    INFERENCE_BENCHMARK_PATH,
    INPUT_FEATURES,
    METRICS_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_METADATA_PATH,
)
from src.prediction import load_metadata, load_model, predict_stability
from src.visualization import (
    PCA_DISCLAIMER,
    compute_pca_projection,
    plot_correlation_matrix,
    plot_feature_boxplots,
    plot_feature_distribution,
    plot_kernel_comparison,
    plot_pca_scatter,
    plot_target_distribution,
)

# Set page config
st.set_page_config(
    page_title="Electrical Grid Stability Classification",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=5)
def load_all_artifacts():
    """Load cached artifacts, metrics, and metadata with auto-refresh."""
    metadata = {}
    if MODEL_METADATA_PATH.exists() and MODEL_METADATA_PATH.stat().st_size > 0:
        try:
            with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            st.error(f"Error loading metadata: {e}")

    metrics = {}
    if METRICS_PATH.exists() and METRICS_PATH.stat().st_size > 0:
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception as e:
            st.error(f"Error loading metrics: {e}")

    comparison_df = None
    if MODEL_COMPARISON_PATH.exists() and MODEL_COMPARISON_PATH.stat().st_size > 0:
        try:
            comparison_df = pd.read_csv(MODEL_COMPARISON_PATH)
        except Exception as e:
            st.error(f"Error loading model comparison: {e}")

    importance_df = None
    if FEATURE_IMPORTANCE_PATH.exists() and FEATURE_IMPORTANCE_PATH.stat().st_size > 0:
        try:
            importance_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
        except Exception as e:
            st.error(f"Error loading feature importance: {e}")

    benchmark = {}
    if INFERENCE_BENCHMARK_PATH.exists() and INFERENCE_BENCHMARK_PATH.stat().st_size > 0:
        try:
            with open(INFERENCE_BENCHMARK_PATH, "r", encoding="utf-8") as f:
                benchmark = json.load(f)
        except Exception as e:
            st.error(f"Error loading inference benchmark: {e}")

    raw_data = None
    if DATASET_RAW_PATH.exists() and DATASET_RAW_PATH.stat().st_size > 0:
        try:
            raw_data = pd.read_csv(DATASET_RAW_PATH)
        except Exception as e:
            st.error(f"Error loading raw dataset: {e}")

    return metadata, metrics, comparison_df, importance_df, benchmark, raw_data


# Sidebar Reload Artifacts button
if st.sidebar.button("🔄 Reload Model Artifacts", help="Flush cache and reload generated models and artifacts"):
    st.cache_data.clear()
    st.rerun()

metadata, metrics, comparison_df, importance_df, benchmark, raw_data = load_all_artifacts()

# Sidebar Navigation
st.sidebar.title("⚡ Grid Stability AI")
st.sidebar.caption("Decentralized Power System Classification")

nav_selection = st.sidebar.radio(
    "Navigation",
    [
        "1. Overview",
        "2. Stability Prediction",
        "3. Model Comparison",
        "4. Kernel Analysis",
        "5. Feature Importance",
        "6. Data Exploration",
        "7. Inference Benchmark",
        "8. About & Limitations",
    ],
)

st.sidebar.markdown("---")
# Deployment Mode Selector (Section 33)
serving_mode = st.sidebar.selectbox(
    "Prediction Engine Mode",
    ["Local Pipeline", "FastAPI REST Service"],
    help="Toggle between direct in-process inference and calling the FastAPI service endpoint.",
)
api_endpoint_input = API_URL
if serving_mode == "FastAPI REST Service":
    api_endpoint_input = st.sidebar.text_input("FastAPI Base URL", value=API_URL)


# ==============================================================================
# 1. OVERVIEW PAGE
# ==============================================================================
if nav_selection == "1. Overview":
    st.title("⚡ Electrical Grid Stability Classification")
    st.markdown(
        """
        An enterprise-grade, reproducible machine learning system designed to classify the stability 
        of decentralized 4-node electrical power grid architectures based on dynamic reaction times, 
        power flows, and price elasticity coefficients.
        """
    )

    if not metadata:
        st.warning(
            "⚠️ Trained model artifacts not detected. Please run `python train.py` to train all 5 models and generate live artifacts."
        )
    else:
        best_model_name = metadata.get("model_name", "N/A")
        train_n = metadata.get("train_samples", 0)
        test_n = metadata.get("test_samples", 0)
        total_samples = train_n + test_n
        holdout = metadata.get("holdout_performance", {})

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Best Model", best_model_name)
        col2.metric("Holdout Accuracy", f"{holdout.get('accuracy', 0)*100:.2f}%")
        col3.metric("Holdout F1-Score", f"{holdout.get('f1', 0)*100:.2f}%")
        col4.metric("Holdout ROC-AUC", f"{holdout.get('roc_auc', 0)*100:.2f}%" if holdout.get("roc_auc") else "N/A")

        st.markdown("---")
        st.subheader("System Architecture & Validation Workflow")

        col_a, col_b = st.columns([1.5, 1])

        with col_a:
            st.markdown(
                """
                ### Zero-Leakage ML Engineering
                - **Dataset**: UCI Electrical Grid Stability Simulated Dataset (~10,000 observations, 12 dynamic input features).
                - **Target Formulation**: Binary state: `0 = Stable`, `1 = Unstable`. (The continuous root `stab` is completely excluded to prevent target leakage).
                - **Cross-Validation**: Stratified 5-Fold Cross-Validation executed entirely inside scikit-learn `Pipeline` objects.
                - **Isolated Holdout**: 20% test split isolated prior to any fitting or hyperparameter optimization.
                - **Evaluated Models**: Linear SVM, Polynomial SVM, RBF SVM, K-Nearest Neighbors, and Decision Trees.
                """
            )

        with col_b:
            st.markdown("### Operational Class Summary")
            if raw_data is not None and "stabf" in raw_data.columns:
                fig_target = plot_target_distribution(raw_data["stabf"])
                st.plotly_chart(fig_target, use_container_width=True)

        st.info(
            "💡 **Scientific Note**: Features reflect simulation outputs from dynamical swing equation models. "
            "Model outputs reflect statistical relationships within the simulation domain and do not replace physical protection relays."
        )


# ==============================================================================
# 2. STABILITY PREDICTION PAGE
# ==============================================================================
elif nav_selection == "2. Stability Prediction":
    st.title("🎯 Real-Time Grid Stability Prediction")
    st.markdown("Simulate a decentralized grid state and compute synchronization stability.")

    ranges = metadata.get("feature_ranges", {}) if metadata else {}

    def get_slider_bounds(feature, default_min, default_max, default_val):
        if feature in ranges:
            f_min = float(ranges[feature]["min"])
            f_max = float(ranges[feature]["max"])
            f_mean = float(ranges[feature]["mean"])
            return f_min, f_max, f_mean
        return default_min, default_max, default_val

    st.subheader("1. Participant Reaction Times (τ1 – τ4)")
    st.caption("Time required for participants to adapt their generation/consumption to price changes (seconds).")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        mi, ma, me = get_slider_bounds("tau1", 0.5, 10.0, 2.959)
        tau1 = st.slider("τ1 (Generator)", mi, ma, me, step=0.01)
    with c2:
        mi, ma, me = get_slider_bounds("tau2", 0.5, 10.0, 3.080)
        tau2 = st.slider("τ2 (Consumer 1)", mi, ma, me, step=0.01)
    with c3:
        mi, ma, me = get_slider_bounds("tau3", 0.5, 10.0, 8.381)
        tau3 = st.slider("τ3 (Consumer 2)", mi, ma, me, step=0.01)
    with c4:
        mi, ma, me = get_slider_bounds("tau4", 0.5, 10.0, 9.781)
        tau4 = st.slider("τ4 (Consumer 3)", mi, ma, me, step=0.01)

    st.subheader("2. Power Flow Parameters (p1 – p4)")
    st.caption("Net power generated (positive, p1) and consumed (negative, p2..p4). Nominal balance: p1 + p2 + p3 + p4 = 0.")
    p1_col, p2_col, p3_col, p4_col = st.columns(4)
    with p1_col:
        mi, ma, me = get_slider_bounds("p1", 0.5, 6.0, 3.763)
        p1 = st.slider("p1 (Generator Net)", mi, ma, me, step=0.01)
    with p2_col:
        mi, ma, me = get_slider_bounds("p2", -2.0, -0.5, -1.527)
        p2 = st.slider("p2 (Consumer 1)", mi, ma, me, step=0.01)
    with p3_col:
        mi, ma, me = get_slider_bounds("p3", -2.0, -0.5, -1.390)
        p3 = st.slider("p3 (Consumer 2)", mi, ma, me, step=0.01)
    with p4_col:
        mi, ma, me = get_slider_bounds("p4", -2.0, -0.5, -0.845)
        p4 = st.slider("p4 (Consumer 3)", mi, ma, me, step=0.01)

    st.subheader("3. Price Elasticity Coefficients (g1 – g4)")
    st.caption("Price sensitivity parameters determining participant demand response.")
    g1_col, g2_col, g3_col, g4_col = st.columns(4)
    with g1_col:
        mi, ma, me = get_slider_bounds("g1", 0.05, 1.0, 0.562)
        g1 = st.slider("γ1 (Elasticity 1)", mi, ma, me, step=0.01)
    with g2_col:
        mi, ma, me = get_slider_bounds("g2", 0.05, 1.0, 0.413)
        g2 = st.slider("γ2 (Elasticity 2)", mi, ma, me, step=0.01)
    with g3_col:
        mi, ma, me = get_slider_bounds("g3", 0.05, 1.0, 0.778)
        g3 = st.slider("γ3 (Elasticity 3)", mi, ma, me, step=0.01)
    with g4_col:
        mi, ma, me = get_slider_bounds("g4", 0.05, 1.0, 0.958)
        g4 = st.slider("γ4 (Elasticity 4)", mi, ma, me, step=0.01)

    payload = {
        "tau1": tau1,
        "tau2": tau2,
        "tau3": tau3,
        "tau4": tau4,
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p4": p4,
        "g1": g1,
        "g2": g2,
        "g3": g3,
        "g4": g4,
    }

    net_power = p1 + p2 + p3 + p4
    st.caption(f"System Net Power Balance: `{net_power:+.4f}`")

    st.markdown("---")
    predict_btn = st.button("⚡ Predict Grid Stability", type="primary", use_container_width=True)

    if predict_btn:
        try:
            if serving_mode == "FastAPI REST Service":
                st.info(f"Connecting to REST endpoint `{api_endpoint_input}/predict`...")
                resp = requests.post(f"{api_endpoint_input}/predict", json=payload, timeout=5)
                if resp.status_code == 200:
                    result = resp.json()
                else:
                    st.error(f"API Error ({resp.status_code}): {resp.text}")
                    st.stop()
            else:
                result = predict_stability(payload)

            is_stable = result["prediction"] == 0
            label = result["label"].upper()
            prob_stable = result["probability_stable"]
            prob_unstable = result["probability_unstable"]

            st.markdown("### Prediction Outcome")
            res_col1, res_col2 = st.columns([1, 1.5])

            with res_col1:
                if is_stable:
                    st.success(f"### Grid State: **{label}** (Class 0)")
                    st.markdown("The decentralized grid is predicted to maintain synchronization stability.")
                else:
                    st.error(f"### Grid State: **{label}** (Class 1)")
                    st.markdown("The decentralized grid is predicted to exhibit synchronization instability.")

                st.write(f"**Model Engine**: `{result.get('model', 'Ensemble')}`")
                st.write(f"**Serving Mode**: `{serving_mode}`")

            with res_col2:
                st.markdown("**Predicted Probability Distribution:**")
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=prob_unstable * 100,
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": "Instability Probability (%)", "font": {"size": 18}},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#d62728" if not is_stable else "#2ca02c"},
                            "steps": [
                                {"range": [0, 50], "color": "#e8f5e9"},
                                {"range": [50, 100], "color": "#ffebee"},
                            ],
                            "threshold": {
                                "line": {"color": "black", "width": 4},
                                "thickness": 0.75,
                                "value": 50,
                            },
                        },
                    )
                )
                fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

        except Exception as e:
            st.error(f"Inference Failure: {e}")


# ==============================================================================
# 3. MODEL COMPARISON PAGE
# ==============================================================================
elif nav_selection == "3. Model Comparison":
    st.title("📊 Multi-Model Benchmark Comparison")
    st.markdown(
        """
        Empirical evaluation of 5 distinct classification architectures on the 
        Electrical Grid Stability dataset using **Stratified 5-Fold Cross-Validation** 
        and an **Isolated 20% Holdout Test Set**.
        """
    )

    if comparison_df is not None:
        st.subheader("Model Performance Summary Table")
        display_df = comparison_df.copy()
        best_f1 = display_df["cv_f1_mean"].max()

        def highlight_best(row):
            return ["background-color: #d4edda; font-weight: bold" if row["cv_f1_mean"] == best_f1 else "" for _ in row]

        st.dataframe(
            display_df[[
                "model", "cv_f1_mean", "cv_accuracy_mean", "test_f1", 
                "test_accuracy", "test_precision", "test_recall", "training_time_sec"
            ]].rename(columns={
                "model": "Model",
                "cv_f1_mean": "CV F1 (Mean)",
                "cv_accuracy_mean": "CV Accuracy",
                "test_f1": "Test F1",
                "test_accuracy": "Test Accuracy",
                "test_precision": "Test Precision",
                "test_recall": "Test Recall",
                "training_time_sec": "Train Time (s)",
            }),
            use_container_width=True,
        )

        st.markdown("---")
        st.subheader("Holdout Metric Comparison by Model")
        fig_comp = px.bar(
            comparison_df,
            x="model",
            y=["test_accuracy", "test_f1", "test_precision", "test_recall"],
            barmode="group",
            title="Model Holdout Performance Metrics",
            labels={"value": "Score", "model": "Model", "variable": "Metric"},
            color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
        )
        fig_comp.update_layout(yaxis=dict(range=[0.75, 1.0]), template="plotly_white")
        st.plotly_chart(fig_comp, use_container_width=True)

        st.subheader("Holdout Confusion Matrix")
        if CONFUSION_MATRIX_PATH.exists():
            st.image(str(CONFUSION_MATRIX_PATH), caption="Confusion Matrix on Independent Holdout Test Set", width=550)
    else:
        st.warning("Comparison artifacts not yet generated. Please execute `python train.py`.")


# ==============================================================================
# 4. KERNEL ANALYSIS PAGE
# ==============================================================================
elif nav_selection == "4. Kernel Analysis":
    st.title("🔬 Support Vector Machine Kernel Comparison")
    st.markdown(
        """
        In-depth comparison of **Linear**, **Polynomial**, and **Radial Basis Function (RBF)** 
        kernels to evaluate how kernel geometry captures the underlying physical dynamical relationships.
        """
    )

    if comparison_df is not None:
        svm_df = comparison_df[comparison_df["model"].str.contains("SVM")].copy()
        svm_df["kernel"] = svm_df["model"].str.replace(" SVM", "")

        k_col1, k_col2 = st.columns(2)
        with k_col1:
            fig_svm = plot_kernel_comparison(svm_df)
            st.plotly_chart(fig_svm, use_container_width=True)

        with k_col2:
            fig_time = px.bar(
                svm_df,
                x="kernel",
                y="training_time_sec",
                title="Training Computational Time (seconds)",
                labels={"kernel": "SVM Kernel", "training_time_sec": "Time (s)"},
                color="kernel",
                color_discrete_sequence=["#636EFA", "#EF553B", "#00CC96"],
            )
            fig_time.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig_time, use_container_width=True)

        st.markdown("---")
        st.subheader("Mathematical and Physical Interpretation")
        t1, t2, t3 = st.tabs(["Linear Kernel", "Polynomial Kernel", "RBF Kernel"])

        with t1:
            st.markdown(
                """
                ### Linear SVM (`kernel='linear'`)
                - **Decision Boundary Formulation**: Hyperplane in the original feature space:
                  $$\\mathbf{w}^T \\mathbf{x} + b = 0$$
                - **Behavior**: Separates classes using linear combinations of grid parameters.
                - **Observation**: Power system stability equations involve nonlinear trigonometric functions 
                  (swing equation $\\ddot{\\delta} \\propto \\sin(\\delta)$) and quadratic power flows. 
                  A linear boundary achieves reasonable baseline separation but cannot capture multi-participant 
                  dynamic phase interactions.
                """
            )

        with t2:
            st.markdown(
                """
                ### Polynomial SVM (`kernel='poly'`)
                - **Kernel Function**:
                  $$K(\\mathbf{x}, \\mathbf{x}') = (\\gamma \\mathbf{x}^T \\mathbf{x}' + c)^d$$
                - **Behavior**: Projects inputs into degree-$d$ monomial feature spaces, explicitly modeling 
                  multi-way interactions like $p_i \\cdot g_i$ and reaction-time products.
                - **Trade-off**: Higher computational training cost with cubic kernel matrix calculations, 
                  achieving strong separation of polynomial interaction terms.
                """
            )

        with t3:
            st.markdown(
                """
                ### Radial Basis Function SVM (`kernel='rbf'`)
                - **Kernel Function**:
                  $$K(\\mathbf{x}, \\mathbf{x}') = \\exp(-\\gamma \\|\\mathbf{x} - \\mathbf{x}'\\|^2)$$
                - **Behavior**: Maps inputs into an infinite-dimensional Hilbert space, forming localized, 
                  smooth, highly adaptive decision regions.
                - **Observation**: RBF kernel captures localized pockets of stability where complex combinations 
                  of participant reaction delays and price elasticity yield stable synchronization.
                """
            )


# ==============================================================================
# 5. FEATURE IMPORTANCE PAGE
# ==============================================================================
elif nav_selection == "5. Feature Importance":
    st.title("🔍 Model Explainability & Feature Importance")
    st.markdown(
        """
        Permutation feature importance calculated on the isolated holdout test set. 
        This assesses how heavily model predictions degrade when each individual electrical parameter is shuffled.
        """
    )

    if importance_df is not None:
        fig_imp = px.bar(
            importance_df.sort_values(by="importance_mean", ascending=True),
            x="importance_mean",
            y="feature",
            error_x="importance_std",
            orientation="h",
            title="Permutation Feature Importance (Decrease in Holdout F1)",
            labels={"importance_mean": "Mean Decrease in F1", "feature": "Parameter"},
            color="importance_mean",
            color_continuous_scale="Blues",
        )
        fig_imp.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig_imp, use_container_width=True)

        st.warning(
            "⚠️ **Methodological & Physical Integrity Disclaimer**: "
            "Feature importance reflects model behavior and does not establish physical causality. "
            "Parameters with high permutation importance indicate strong predictive sensitivity in the "
            "classifier, not direct mechanistic necessity for real-world grid control."
        )

        st.markdown("### Top Ranked Model Features")
        st.dataframe(importance_df, use_container_width=True)
    else:
        st.warning("Feature importance artifacts not found. Please run `python train.py`.")


# ==============================================================================
# 6. DATA EXPLORATION PAGE
# ==============================================================================
elif nav_selection == "6. Data Exploration":
    st.title("📈 Exploratory Data Analysis")
    st.markdown("Examine distributions, correlations, and relationships among electrical parameters.")

    if raw_data is not None:
        sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
            "Feature Distributions", "Boxplot Comparisons", "Correlation Matrix", "2D PCA Projection"
        ])

        with sub_tab1:
            st.subheader("Univariate Distribution by Stability Class")
            feat_select = st.selectbox("Select Parameter to Inspect", INPUT_FEATURES)
            fig_dist = plot_feature_distribution(raw_data, feat_select)
            st.plotly_chart(fig_dist, use_container_width=True)

        with sub_tab2:
            st.subheader("Parameter Group Boxplots")
            grp_select = st.radio("Parameter Group", ["tau", "power", "elasticity"], horizontal=True)
            fig_box = plot_feature_boxplots(raw_data, grp_select)
            st.plotly_chart(fig_box, use_container_width=True)

        with sub_tab3:
            st.subheader("Parameter Correlation Structure")
            fig_corr = plot_correlation_matrix(raw_data)
            st.plotly_chart(fig_corr, use_container_width=True)

        with sub_tab4:
            st.subheader("2D PCA Visualization of Parameter Space")
            from src.validation import validate_raw_dataset
            X_clean, y_clean, _ = validate_raw_dataset(raw_data)
            pca_df, pca_model = compute_pca_projection(X_clean, y_clean)
            fig_pca = plot_pca_scatter(pca_df, pca_model)
            st.plotly_chart(fig_pca, use_container_width=True)
            st.caption(f"📌 {PCA_DISCLAIMER}")
    else:
        st.warning("Raw dataset file not found. Ingest the dataset by running `python train.py`.")


# ==============================================================================
# 7. INFERENCE BENCHMARK PAGE
# ==============================================================================
elif nav_selection == "7. Inference Benchmark":
    st.title("⚡ Inference Latency & Throughput Benchmark")
    st.markdown(
        """
        High-precision latency profiling executed with `time.perf_counter()` 
        measuring end-to-end classification latency across single-sample and batch queries.
        """
    )

    if benchmark:
        single = benchmark.get("single_sample", {})
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        col_b1.metric("Mean Latency", f"{single.get('mean_ms', 0):.4f} ms")
        col_b2.metric("Median Latency", f"{single.get('median_ms', 0):.4f} ms")
        col_b3.metric("P95 Latency", f"{single.get('p95_ms', 0):.4f} ms")
        col_b4.metric("Throughput", f"{single.get('throughput_samples_per_sec', 0):.0f} /sec")

        st.markdown("---")
        st.subheader("Batch Scaling Latency Breakdown")
        batch_data = benchmark.get("batch_benchmarks", {})
        if batch_data:
            b_df = pd.DataFrame(batch_data.values())
            st.dataframe(b_df.rename(columns={
                "batch_size": "Batch Size",
                "mean_total_ms": "Total Latency (ms)",
                "latency_per_sample_ms": "Latency per Sample (ms)",
                "samples_per_second": "Throughput (samples/s)",
            }), use_container_width=True)

            fig_lat = px.line(
                b_df,
                x="batch_size",
                y="latency_per_sample_ms",
                markers=True,
                title="Per-Sample Latency Scaling by Batch Size",
                labels={"batch_size": "Batch Size", "latency_per_sample_ms": "Latency / Sample (ms)"},
            )
            fig_lat.update_layout(template="plotly_white")
            st.plotly_chart(fig_lat, use_container_width=True)

        st.subheader("Hardware & Benchmark Context")
        plat = benchmark.get("platform", {})
        st.json(plat)
        st.info(
            "📌 **Scientific Clarification**: The measured inference latency demonstrates the computational "
            "feasibility of low-latency classification under the benchmark conditions."
        )
    else:
        st.warning("Benchmark artifact not found. Execute `python benchmark.py` or `python train.py`.")


# ==============================================================================
# 8. ABOUT & LIMITATIONS PAGE
# ==============================================================================
elif nav_selection == "8. About & Limitations":
    st.title("ℹ️ About the Project & Scientific Limitations")

    st.markdown(
        """
        ### Project Background
        This project investigates machine learning approaches for decentralized smart-grid stability 
        synchronization. In decentralized systems, consumer and producer participants interact dynamically, 
        adjusting generation and consumption based on real-time price elasticity. Synchronization stability 
        is governed by coupled nonlinear differential equations.

        ### Citation & Dataset Provenance
        - **Dataset**: Electrical Grid Stability Simulated Data Set (UCI Machine Learning Repository, ID: 471)
        - **Authors**: Vadim Arzamasov, Klemens Böhm, Patrick Jochem (Karlsruhe Institute of Technology)
        - **Simulated Architecture**: 4-node star decentralized power grid (1 producer, 3 consumers).

        ---
        ### Formal Scientific Limitations & Engineering Disclaimers

        > ⚠️ **CRITICAL ENGINEERING DISCLAIMER**  
        > **This model should not be used for operational grid control without extensive domain validation, 
        > safety analysis, and regulatory/engineering approval.**

        1. **Simulation-Based Nature**: The dataset comprises synthetic simulations from a simplified 4-node 
           mathematical model. Real-world power networks encompass thousands of interconnected nodes, non-sinusoidal 
           harmonics, reactive power considerations, and stochastic line faults.
        2. **Domain Shift Risk**: Pure simulation-trained models risk catastrophic domain shift when exposed to 
           physical substation telemetry containing sensor noise, packet loss, and unmodeled impedances.
        3. **Absence of Field Validation**: No live physical grid or microgrid hardware-in-the-loop (HIL) testing 
           has been performed.
        4. **Computational Scaling of SVMs**: While RBF SVM demonstrates high accuracy on 10,000 samples, its quadratic-to-cubic 
           training complexity $\\mathcal{O}(N^2 \\text{ to } N^3)$ limits direct scaling to millions of live telemetry streams.
        5. **Correlation vs Causality**: The model identifies parameter regions associated with higher predicted 
           instability within the simulation dataset. Feature importance measures classifier dependency, not physical causality.
        """
    )

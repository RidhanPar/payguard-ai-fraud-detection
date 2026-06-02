"""
PayGuard Streamlit Dashboard.

A professional multi-page fraud detection dashboard for the PayGuard project.

Run from the project root:

    streamlit run app/dashboard.py

Expected project files:

    models/fraud_model.pkl
    models/scaler.pkl
    data/processed/X_test.pkl
    data/processed/y_test.pkl
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import streamlit as st

st.set_page_config(
    page_title="PayGuard | AI Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


REQUIRED_PACKAGES: Dict[str, str] = {
    "pandas": "pandas",
    "numpy": "numpy",
    "plotly": "plotly",
    "joblib": "joblib",
    "sklearn": "scikit-learn",
}

OPTIONAL_PACKAGES: Dict[str, str] = {
    "shap": "shap",
    "matplotlib": "matplotlib",
}


def check_requirements() -> None:
    """Check that required dashboard dependencies are installed."""
    missing_required = [
        pip_name
        for import_name, pip_name in REQUIRED_PACKAGES.items()
        if importlib.util.find_spec(import_name) is None
    ]

    missing_optional = [
        pip_name
        for import_name, pip_name in OPTIONAL_PACKAGES.items()
        if importlib.util.find_spec(import_name) is None
    ]

    if missing_required:
        st.error(
            "Missing required packages: "
            + ", ".join(missing_required)
            + ". Run: `python -m pip install -r requirements.txt`"
        )
        st.stop()

    if missing_optional:
        st.warning(
            "Optional explainability packages are missing: "
            + ", ".join(missing_optional)
            + ". The dashboard will work, but SHAP charts may be unavailable."
        )


check_requirements()


import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    import matplotlib.pyplot as plt
    import shap

    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
X_TEST_PATH = PROCESSED_DIR / "X_test.pkl"
Y_TEST_PATH = PROCESSED_DIR / "y_test.pkl"


CUSTOM_CSS = """
<style>
    html, body, [class*="css"] {
        font-family: "Inter", "Segoe UI", sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 2rem 2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(26, 26, 46, 0.20);
    }

    .logo-text {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin-bottom: 0.2rem;
    }

    .logo-subtitle {
        color: #d7fbe8;
        font-size: 1rem;
        margin-top: 0.2rem;
    }

    .metric-card {
        background: #ffffff;
        padding: 1.25rem;
        border-radius: 16px;
        border: 1px solid #edf0f5;
        box-shadow: 0 6px 20px rgba(16, 24, 40, 0.07);
        min-height: 120px;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #667085;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }

    .metric-value {
        font-size: 1.8rem;
        color: #1a1a2e;
        font-weight: 800;
        line-height: 1.2;
    }

    .metric-help {
        font-size: 0.78rem;
        color: #98a2b3;
        margin-top: 0.4rem;
    }

    .risk-card {
        padding: 1rem;
        border-radius: 14px;
        margin-bottom: 0.75rem;
        border: 1px solid #edf0f5;
        box-shadow: 0 3px 12px rgba(16, 24, 40, 0.06);
        background: white;
    }

    .factor-title {
        font-weight: 750;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }

    .factor-body {
        color: #475467;
        font-size: 0.9rem;
    }

    .stButton > button {
        background-color: #1a1a2e;
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: 700;
        padding: 0.6rem 1rem;
    }

    .stDownloadButton > button {
        background-color: #0f9d58;
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: 700;
        padding: 0.6rem 1rem;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def format_currency(amount: float, currency: str = "£") -> str:
    """Format a numeric amount as currency."""
    try:
        return f"{currency}{float(amount):,.2f}"
    except Exception:
        return f"{currency}0.00"


def show_header(title: str, subtitle: str) -> None:
    """Render the PayGuard header."""
    st.markdown(
        f"""
        <div class="main-header">
            <div class="logo-text">🛡️ PayGuard</div>
            <div class="logo-subtitle">{title} — {subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, help_text: str = "") -> None:
    """Render a styled metric card."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def model_not_trained_warning() -> None:
    """Show warning and instructions when model artifacts are missing."""
    st.warning("Model not trained yet. Please run the training pipeline first.")
    st.markdown("""
        Required files:

        ```text
        models/fraud_model.pkl
        models/scaler.pkl
        ```

        Run these notebooks in order:

        ```powershell
        python -m notebook notebooks/02_preprocessing.ipynb
        python -m notebook notebooks/03_modelling.ipynb
        ```

        Then restart the dashboard:

        ```powershell
        streamlit run app/dashboard.py
        ```
        """)


@st.cache_resource(show_spinner=False)
def load_predictor() -> Optional[Any]:
    """Load FraudPredictor from src/predict.py."""
    try:
        if not MODEL_PATH.exists() or not SCALER_PATH.exists():
            return None

        from src.predict import FraudPredictor

        return FraudPredictor(model_path=MODEL_PATH, scaler_path=SCALER_PATH)
    except Exception as exc:
        logger.exception("Failed to load FraudPredictor.")
        st.error(f"Could not load predictor: {exc}")
        return None


@st.cache_resource(show_spinner=False)
def load_model() -> Optional[Any]:
    """Load trained fraud model."""
    try:
        if not MODEL_PATH.exists():
            return None
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        logger.exception("Failed to load model.")
        st.error(f"Could not load model: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_test_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Load processed test data."""
    try:
        if not X_TEST_PATH.exists() or not Y_TEST_PATH.exists():
            return None, None

        X_test = joblib.load(X_TEST_PATH)
        y_test = joblib.load(Y_TEST_PATH)

        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)

        if not isinstance(y_test, pd.Series):
            y_test = pd.Series(y_test, index=X_test.index, name="Class")

        return X_test, y_test
    except Exception as exc:
        logger.exception("Failed to load test data.")
        st.error(f"Could not load test data: {exc}")
        return None, None


def risk_level_color(risk_level: str) -> str:
    """Return color for a risk level."""
    colors = {
        "LOW": "#0f9d58",
        "MEDIUM": "#fbbc04",
        "HIGH": "#f97316",
        "CRITICAL": "#d93025",
    }
    return colors.get(risk_level, "#667085")


def styled_risk_table(df: pd.DataFrame) -> Any:
    """Apply color styling to risk-level results table."""

    def style_risk(value: Any) -> str:
        color_map = {
            "LOW": "background-color: #e8f8ef; color: #027a48; font-weight: 700;",
            "MEDIUM": "background-color: #fff7d6; color: #92400e; font-weight: 700;",
            "HIGH": "background-color: #ffedd5; color: #c2410c; font-weight: 700;",
            "CRITICAL": "background-color: #fee2e2; color: #b42318; font-weight: 700;",
        }
        return color_map.get(str(value), "")

    if "risk_level" in df.columns:
        return df.style.map(style_risk, subset=["risk_level"])
    return df.style


def create_demo_transactions(predictor: Any, n: int = 500) -> pd.DataFrame:
    """Create sample predictions from processed test data for dashboard charts."""
    X_test, y_test = load_test_data()

    if X_test is None:
        return pd.DataFrame()

    sample = X_test.sample(n=min(n, len(X_test)), random_state=42).copy()
    predictions = predictor.predict_batch(sample)

    if y_test is not None:
        predictions["Class"] = y_test.loc[sample.index].values

    predictions["transaction_time"] = pd.date_range(
        start="2026-01-01",
        periods=len(predictions),
        freq="h",
    )

    predictions["Amount"] = np.random.default_rng(42).lognormal(
        mean=3.5,
        sigma=1.0,
        size=len(predictions),
    )

    return predictions.reset_index(drop=True)


def get_positive_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return positive fraud-class scores."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))

    raise AttributeError("Model must support predict_proba or decision_function.")


def calculate_metrics_at_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Calculate classification metrics at a custom threshold."""
    y_pred = (probabilities >= threshold).astype(int)

    return {
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, probabilities),
        "Accuracy": accuracy_score(y_true, y_pred),
    }


def plain_english_factor(feature: str, value: float) -> str:
    """Convert model feature name into plain-English explanation."""
    if feature == "Amount_Scaled":
        return "The transaction amount is unusual compared with normal payment values."
    if feature == "Time_Scaled":
        return "The transaction timing pattern is different from typical payment activity."
    if feature.startswith("V"):
        return (
            f"{feature} relates to the transaction signature pattern learned from "
            "anonymized payment behavior. A strong value can indicate fraud-like activity."
        )
    return "This feature contributed to the model's fraud risk score."


def page_overview(predictor: Any) -> None:
    """Render dashboard overview page."""
    show_header(
        "AI-Powered Payment Fraud Detection",
        "real-time risk monitoring for transaction teams",
    )

    demo_df = create_demo_transactions(predictor, n=500)

    if demo_df.empty:
        st.info("No processed sample data found. Run preprocessing first to populate charts.")
        return

    total_transactions = len(demo_df)
    fraud_detected = int(demo_df["is_fraud"].sum())
    high_risk_df = demo_df[demo_df["risk_level"].isin(["HIGH", "CRITICAL"])]
    amount_protected = float(high_risk_df.get("Amount", pd.Series(dtype=float)).sum())

    model_accuracy = "N/A"
    if "Class" in demo_df.columns:
        model_accuracy = (
            f"{accuracy_score(demo_df['Class'], demo_df['is_fraud'].astype(int)) * 100:.2f}%"
        )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card(
            "Total Transactions Analysed",
            f"{total_transactions:,}",
            "Sample transactions processed by PayGuard",
        )
    with col2:
        metric_card(
            "Fraud Detected", f"{fraud_detected:,}", "Transactions flagged as potential fraud"
        )
    with col3:
        metric_card(
            "Amount Protected",
            format_currency(amount_protected),
            "Estimated value protected from high-risk transactions",
        )
    with col4:
        metric_card(
            "Model Accuracy", model_accuracy, "Calculated from available labelled test data"
        )

    st.markdown("### Fraud Monitoring Trends")
    chart_col1, chart_col2 = st.columns([2, 1])

    with chart_col1:
        trend_df = (
            demo_df.assign(fraud_flag=demo_df["is_fraud"].astype(int))
            .set_index("transaction_time")
            .resample("12h")["fraud_flag"]
            .sum()
            .reset_index()
        )

        fig = px.line(
            trend_df,
            x="transaction_time",
            y="fraud_flag",
            markers=True,
            title="Fraud Detections Over Time",
            labels={"transaction_time": "Time", "fraud_flag": "Fraud Detections"},
        )
        fig.update_layout(template="plotly_white", title_x=0.02)
        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        risk_distribution = (
            demo_df["risk_level"]
            .value_counts()
            .reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"])
            .fillna(0)
            .reset_index()
        )
        risk_distribution.columns = ["risk_level", "count"]

        fig = px.pie(
            risk_distribution,
            names="risk_level",
            values="count",
            title="Risk Level Distribution",
            hole=0.45,
            color="risk_level",
            color_discrete_map={
                "LOW": "#0f9d58",
                "MEDIUM": "#fbbc04",
                "HIGH": "#f97316",
                "CRITICAL": "#d93025",
            },
        )
        fig.update_layout(template="plotly_white", title_x=0.02)
        st.plotly_chart(fig, use_container_width=True)


def page_analyse_transactions(predictor: Any) -> None:
    """Render transaction analysis page."""
    show_header("Analyse Transactions", "upload a CSV and run AI-powered fraud scoring")

    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"],
        help="CSV should include V1-V28 and either Amount/Time or Amount_Scaled/Time_Scaled.",
    )

    if uploaded_file is None:
        st.info("Upload a transaction CSV to begin analysis.")
        return

    try:
        uploaded_df = pd.read_csv(uploaded_file)
        st.session_state["uploaded_df"] = uploaded_df

        st.markdown("### Uploaded Data Preview")
        st.dataframe(uploaded_df.head(), use_container_width=True)

        if st.button("Run Fraud Analysis", type="primary"):
            with st.spinner("Running fraud analysis..."):
                results_df = predictor.predict_batch(uploaded_df)
                st.session_state["analysis_results"] = results_df

        if "analysis_results" in st.session_state:
            results_df = st.session_state["analysis_results"]

            st.markdown("### Fraud Analysis Results")
            st.dataframe(styled_risk_table(results_df.head(100)), use_container_width=True)

            total_analysed = len(results_df)
            high_risk_mask = results_df["risk_level"].isin(["HIGH", "CRITICAL"])
            high_risk_count = int(high_risk_mask.sum())

            if "Amount" in results_df.columns:
                amount_at_risk = float(results_df.loc[high_risk_mask, "Amount"].sum())
            else:
                amount_at_risk = 0.0

            stat_col1, stat_col2, stat_col3 = st.columns(3)
            with stat_col1:
                metric_card("Transactions Analysed", f"{total_analysed:,}")
            with stat_col2:
                metric_card("High-Risk Flagged", f"{high_risk_count:,}")
            with stat_col3:
                metric_card("Amount at Risk", format_currency(amount_at_risk))

            csv_data = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Results as CSV",
                data=csv_data,
                file_name="payguard_fraud_analysis_results.csv",
                mime="text/csv",
            )

    except Exception as exc:
        logger.exception("Transaction analysis failed.")
        st.error(f"Could not analyse uploaded file: {exc}")


def page_model_explainability(predictor: Any) -> None:
    """Render model explainability page."""
    show_header("Model Explainability", "understand why a transaction was flagged")
    st.markdown("## Why did the model flag this transaction?")

    if "analysis_results" not in st.session_state or "uploaded_df" not in st.session_state:
        st.info(
            "Upload and analyse transactions on the 🔍 Analyse Transactions page first. "
            "Then return here to explain a selected transaction."
        )
        return

    uploaded_df = st.session_state["uploaded_df"]
    results_df = st.session_state["analysis_results"]

    selected_transaction_id = st.selectbox(
        "Select Transaction ID",
        options=list(results_df.index.astype(str)),
    )

    selected_index = int(selected_transaction_id)
    selected_raw = uploaded_df.iloc[[selected_index]].copy()

    try:
        processed_instance = predictor.preprocess_input(selected_raw)
        prediction_result = predictor.predict_single(selected_raw.iloc[0].to_dict())

        probability = prediction_result["fraud_probability"]
        risk_level = prediction_result["risk_level"]

        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card("Fraud Probability", f"{probability * 100:.2f}%")
        with col2:
            metric_card("Risk Level", risk_level)
        with col3:
            metric_card("Decision", "Fraud" if prediction_result["is_fraud"] else "Legitimate")

        st.markdown("### SHAP Waterfall Chart")

        if not SHAP_AVAILABLE:
            st.warning(
                "SHAP or Matplotlib is not installed. Run: `python -m pip install shap matplotlib`"
            )
        else:
            with st.spinner("Generating SHAP explanation..."):
                explainer = shap.TreeExplainer(predictor.model)
                local_values = explainer.shap_values(processed_instance)

                if isinstance(local_values, list):
                    local_values = local_values[1]

                expected_value = explainer.expected_value
                if isinstance(expected_value, (list, np.ndarray)):
                    expected_value = expected_value[-1]

                explanation = shap.Explanation(
                    values=np.asarray(local_values)[0],
                    base_values=expected_value,
                    data=processed_instance.iloc[0].values,
                    feature_names=processed_instance.columns.tolist(),
                )

                fig = plt.figure(figsize=(10, 6))
                shap.plots.waterfall(explanation, max_display=10, show=False)
                st.pyplot(fig, clear_figure=True)

        st.markdown("### Top 5 Risk Factors")

        for factor in prediction_result["top_factors"][:5]:
            feature = str(factor.get("feature", "Unknown"))
            value = float(factor.get("value", 0.0))
            impact = float(factor.get("impact_score", 0.0))
            explanation_text = plain_english_factor(feature, value)

            st.markdown(
                f"""
                <div class="risk-card">
                    <div class="factor-title">{feature}</div>
                    <div class="factor-body">
                        Value: <b>{value:.4f}</b> | Impact score: <b>{impact:.4f}</b><br>
                        {explanation_text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    except Exception as exc:
        logger.exception("Explainability page failed.")
        st.error(f"Could not explain selected transaction: {exc}")


def page_model_performance(model: Any) -> None:
    """Render model performance page."""
    show_header("Model Performance", "evaluate fraud detection quality at different thresholds")

    X_test, y_test = load_test_data()

    if X_test is None or y_test is None:
        st.warning(
            "Processed test data not found. Please run `notebooks/02_preprocessing.ipynb` first."
        )
        return

    try:
        probabilities = get_positive_scores(model, X_test)

        threshold = st.slider(
            "Fraud Decision Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="Lower threshold catches more fraud but may increase false positives.",
        )

        metrics = calculate_metrics_at_threshold(y_test, probabilities, threshold)
        y_pred = (probabilities >= threshold).astype(int)

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            metric_card("Precision", f"{metrics['Precision']:.3f}")
        with metric_col2:
            metric_card("Recall", f"{metrics['Recall']:.3f}")
        with metric_col3:
            metric_card("F1", f"{metrics['F1']:.3f}")
        with metric_col4:
            metric_card("AUC", f"{metrics['AUC']:.3f}")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fpr, tpr, _ = roc_curve(y_test, probabilities)
            roc_auc = roc_auc_score(y_test, probabilities)

            roc_fig = go.Figure()
            roc_fig.add_trace(
                go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC AUC = {roc_auc:.4f}")
            )
            roc_fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Random Classifier",
                    line=dict(dash="dash"),
                )
            )
            roc_fig.update_layout(
                title="ROC Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                template="plotly_white",
                title_x=0.02,
            )
            st.plotly_chart(roc_fig, use_container_width=True)

        with chart_col2:
            precision, recall, _ = precision_recall_curve(y_test, probabilities)
            pr_auc = auc(recall, precision)

            pr_fig = go.Figure()
            pr_fig.add_trace(
                go.Scatter(x=recall, y=precision, mode="lines", name=f"PR AUC = {pr_auc:.4f}")
            )
            pr_fig.update_layout(
                title="Precision-Recall Curve",
                xaxis_title="Recall",
                yaxis_title="Precision",
                template="plotly_white",
                title_x=0.02,
            )
            st.plotly_chart(pr_fig, use_container_width=True)

        cm_col, table_col = st.columns([1, 1])

        with cm_col:
            cm = confusion_matrix(y_test, y_pred)
            cm_fig = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title=f"Confusion Matrix at Threshold {threshold:.2f}",
                labels={"x": "Predicted Label", "y": "Actual Label", "color": "Count"},
                x=["Legitimate", "Fraud"],
                y=["Legitimate", "Fraud"],
                color_continuous_scale="Reds",
            )
            cm_fig.update_layout(template="plotly_white", title_x=0.02)
            st.plotly_chart(cm_fig, use_container_width=True)

        with table_col:
            st.markdown("### Metrics Table")
            metrics_df = pd.DataFrame(
                [
                    {
                        "Threshold": threshold,
                        "Precision": metrics["Precision"],
                        "Recall": metrics["Recall"],
                        "F1": metrics["F1"],
                        "AUC": metrics["AUC"],
                        "Accuracy": metrics["Accuracy"],
                    }
                ]
            )
            st.dataframe(metrics_df.round(4), use_container_width=True)

            st.markdown("""
                **How to read this:**

                - Higher **recall** means the model catches more real fraud.
                - Higher **precision** means fewer false fraud alerts.
                - Lowering the threshold usually increases recall but reduces precision.
                """)

    except Exception as exc:
        logger.exception("Model performance page failed.")
        st.error(f"Could not calculate model performance: {exc}")


def main() -> None:
    """Run the Streamlit dashboard."""
    st.sidebar.markdown("## 🛡️ PayGuard")
    st.sidebar.markdown("AI-Powered Payment Fraud Detection")

    page = st.sidebar.radio(
        "Navigation",
        options=[
            "🏠 Overview",
            "🔍 Analyse Transactions",
            "🧠 Model Explainability",
            "📊 Model Performance",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Project Status")

    if MODEL_PATH.exists():
        st.sidebar.success("Model loaded")
    else:
        st.sidebar.error("Model missing")

    if SCALER_PATH.exists():
        st.sidebar.success("Scaler loaded")
    else:
        st.sidebar.error("Scaler missing")

    predictor = load_predictor()
    model = load_model()

    if predictor is None or model is None:
        model_not_trained_warning()
        return

    try:
        if page == "🏠 Overview":
            page_overview(predictor)
        elif page == "🔍 Analyse Transactions":
            page_analyse_transactions(predictor)
        elif page == "🧠 Model Explainability":
            page_model_explainability(predictor)
        elif page == "📊 Model Performance":
            page_model_performance(model)
        else:
            st.error("Unknown page selected.")

    except Exception as exc:
        logger.exception("Unexpected dashboard error.")
        st.error(f"Unexpected dashboard error: {exc}")


if __name__ == "__main__":
    main()

"""
PayGuard Streamlit Dashboard with in-app model training.

This version is fully self-contained for Streamlit Cloud:
- If no trained model exists, the app shows a setup page.
- Users can upload creditcard.csv or use instant demo mode.
- The full training pipeline runs inside Streamlit.
- The trained model is cached in session and saved to disk when possible.
- After training, the app opens the full 4-page fraud detection dashboard.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

st.set_page_config(
    page_title="PayGuard | AI Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

REQUIRED_PACKAGES: Dict[str, str] = {
    "joblib": "joblib",
    "numpy": "numpy",
    "pandas": "pandas",
    "plotly": "plotly",
    "sklearn": "scikit-learn",
    "imblearn": "imbalanced-learn",
    "xgboost": "xgboost",
}


def check_requirements() -> None:
    """Stop the app with a friendly message if a required package is missing."""
    missing = [
        package_name
        for import_name, package_name in REQUIRED_PACKAGES.items()
        if importlib.util.find_spec(import_name) is None
    ]
    if missing:
        st.error("PayGuard cannot start because required packages are missing.")
        st.code("python -m pip install " + " ".join(missing), language="bash")
        st.stop()


check_requirements()

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from imblearn.over_sampling import RandomOverSampler, SMOTE
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

try:
    import matplotlib.pyplot as plt
    import shap

    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports" / "figures"
MODEL_PATH = MODELS_DIR / "fraud_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
TRAINING_ARTIFACT_PATH = MODELS_DIR / "training_artifacts.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.20
DEFAULT_THRESHOLD = 0.50
MAX_UPLOAD_TRAIN_ROWS = 25_000
DEMO_ROWS = 5_000

CUSTOM_CSS = """
<style>
    html, body, [class*="css"] { font-family: "Inter", "Segoe UI", sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 2rem; border-radius: 20px; color: white; margin-bottom: 1.5rem;
        box-shadow: 0 10px 32px rgba(26, 26, 46, 0.22);
    }
    .setup-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem; border-radius: 24px; color: white; margin-bottom: 1.75rem;
        box-shadow: 0 10px 32px rgba(102, 126, 234, 0.25);
    }
    .logo-text { font-size: 2.45rem; font-weight: 850; letter-spacing: -0.05em; }
    .logo-subtitle { color: #eef2ff; font-size: 1.02rem; margin-top: 0.25rem; }
    .metric-card {
        background: #ffffff; padding: 1.25rem; border-radius: 18px; border: 1px solid #edf0f5;
        box-shadow: 0 6px 22px rgba(16, 24, 40, 0.075); min-height: 120px;
    }
    .metric-label { font-size: 0.85rem; color: #667085; font-weight: 650; margin-bottom: 0.45rem; }
    .metric-value { font-size: 1.8rem; color: #1a1a2e; font-weight: 850; line-height: 1.2; }
    .metric-help { font-size: 0.78rem; color: #98a2b3; margin-top: 0.45rem; }
    .risk-card {
        padding: 1rem; border-radius: 14px; margin-bottom: 0.75rem; border: 1px solid #edf0f5;
        box-shadow: 0 3px 12px rgba(16, 24, 40, 0.06); background: white;
    }
    .factor-title { font-weight: 750; color: #1a1a2e; margin-bottom: 0.2rem; }
    .factor-body { color: #475467; font-size: 0.9rem; }
    .success-banner {
        padding: 1rem 1.25rem; background: #ecfdf3; color: #027a48; border: 1px solid #abefc6;
        border-radius: 14px; font-weight: 750; margin-bottom: 1rem;
    }
    .warning-banner {
        padding: 1rem 1.25rem; background: #fffaeb; color: #92400e; border: 1px solid #fedf89;
        border-radius: 14px; font-weight: 650; margin-bottom: 1rem;
    }
    .stButton > button {
        background-color: #1a1a2e; color: white; border-radius: 11px; border: none;
        font-weight: 750; padding: 0.6rem 1rem;
    }
    .stDownloadButton > button {
        background-color: #0f9d58; color: white; border-radius: 11px; border: none;
        font-weight: 750; padding: 0.6rem 1rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def ensure_directories() -> None:
    """Create artifact directories when the filesystem allows it."""
    for directory in [MODELS_DIR, PROCESSED_DIR, REPORTS_DIR]:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning("Could not create directory %s: %s", directory, exc)


def show_header(title: str, subtitle: str) -> None:
    """Render the standard PayGuard dashboard header."""
    st.markdown(
        f"""
        <div class="main-header">
            <div class="logo-text">🛡️ PayGuard</div>
            <div class="logo-subtitle">{title} — {subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_setup_header() -> None:
    """Render the setup page welcome header."""
    st.markdown(
        """
        <div class="setup-header">
            <div class="logo-text">Welcome to PayGuard — Let's set up your model</div>
            <div class="logo-subtitle">
                Upload the Kaggle credit card fraud dataset or launch demo mode
                to train an AI fraud detection model directly inside the app.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, help_text: str = "") -> None:
    """Render a custom metric card."""
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


def success_banner(message: str) -> None:
    """Render a styled success banner."""
    st.markdown(f'<div class="success-banner">{message}</div>', unsafe_allow_html=True)


def format_currency(amount: float, currency: str = "£") -> str:
    """Format a numeric amount as currency."""
    try:
        return f"{currency}{float(amount):,.2f}"
    except Exception:
        return f"{currency}0.00"


def get_risk_level(probability: float) -> str:
    """Convert fraud probability into LOW, MEDIUM, HIGH, or CRITICAL."""
    if probability < 0.30:
        return "LOW"
    if probability < 0.60:
        return "MEDIUM"
    if probability < 0.85:
        return "HIGH"
    return "CRITICAL"


def styled_risk_table(df: pd.DataFrame) -> Any:
    """Apply color styling to a risk-level table."""

    def style_risk(value: Any) -> str:
        styles = {
            "LOW": "background-color: #e8f8ef; color: #027a48; font-weight: 700;",
            "MEDIUM": "background-color: #fff7d6; color: #92400e; font-weight: 700;",
            "HIGH": "background-color: #ffedd5; color: #c2410c; font-weight: 700;",
            "CRITICAL": "background-color: #fee2e2; color: #b42318; font-weight: 700;",
        }
        return styles.get(str(value), "")

    if "risk_level" in df.columns:
        return df.style.map(style_risk, subset=["risk_level"])
    return df.style


def required_creditcard_columns() -> List[str]:
    """Return the expected Kaggle creditcard.csv schema."""
    return ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]


def model_feature_columns() -> List[str]:
    """Return the model input feature order."""
    return [f"V{i}" for i in range(1, 29)] + ["Amount_Scaled", "Time_Scaled"]


def normalise_shap_values(raw_shap_values: Any) -> np.ndarray:
    """Convert SHAP output into a 2D positive-class SHAP array."""
    if isinstance(raw_shap_values, list):
        raw_shap_values = raw_shap_values[-1]
    values = np.asarray(raw_shap_values)
    if values.ndim == 2:
        return values
    if values.ndim == 3:
        if values.shape[-1] <= 2:
            return values[:, :, -1]
        if values.shape[0] <= 2:
            return values[-1, :, :]
    raise ValueError(f"Unsupported SHAP values shape: {values.shape}")


def generate_synthetic_creditcard_data(n_rows: int = DEMO_ROWS, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate synthetic data with the same schema as Kaggle creditcard.csv."""
    rng = np.random.default_rng(seed)
    fraud_rate = 0.02
    n_fraud = max(30, int(n_rows * fraud_rate))
    n_legitimate = n_rows - n_fraud

    legitimate_features = rng.normal(loc=0.0, scale=1.0, size=(n_legitimate, 28))
    fraud_features = rng.normal(loc=0.4, scale=1.25, size=(n_fraud, 28))
    idx = {f"V{i}": i - 1 for i in range(1, 29)}
    fraud_features[:, idx["V14"]] -= rng.normal(4.5, 0.8, size=n_fraud)
    fraud_features[:, idx["V17"]] -= rng.normal(3.8, 0.8, size=n_fraud)
    fraud_features[:, idx["V12"]] -= rng.normal(3.2, 0.7, size=n_fraud)
    fraud_features[:, idx["V10"]] -= rng.normal(2.2, 0.6, size=n_fraud)
    fraud_features[:, idx["V4"]] += rng.normal(2.0, 0.7, size=n_fraud)
    fraud_features[:, idx["V11"]] += rng.normal(1.7, 0.6, size=n_fraud)

    features = np.vstack([legitimate_features, fraud_features])
    target = np.array([0] * n_legitimate + [1] * n_fraud)

    df = pd.DataFrame(features, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", rng.integers(0, 172_800, size=n_rows))
    df["Amount"] = np.concatenate(
        [
            rng.lognormal(mean=3.1, sigma=0.85, size=n_legitimate),
            rng.lognormal(mean=3.7, sigma=1.0, size=n_fraud),
        ]
    )
    df["Class"] = target
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def validate_creditcard_schema(df: pd.DataFrame) -> None:
    """Validate that a dataset matches the creditcard.csv schema."""
    missing = set(required_creditcard_columns()).difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["Class"].nunique() < 2:
        raise ValueError("Class must contain both 0 and 1 values.")
    invalid_classes = set(df["Class"].dropna().unique()).difference({0, 1})
    if invalid_classes:
        raise ValueError("Class must contain only 0 and 1 values.")


def maybe_sample_large_dataset(df: pd.DataFrame, max_rows: int = MAX_UPLOAD_TRAIN_ROWS) -> pd.DataFrame:
    """Stratified sample very large uploads for Streamlit Cloud responsiveness."""
    if len(df) <= max_rows:
        return df
    sampled = (
        df.groupby("Class", group_keys=False)
        .apply(
            lambda group: group.sample(
                n=max(1, int(len(group) / len(df) * max_rows)),
                random_state=RANDOM_STATE,
            )
        )
        .reset_index(drop=True)
    )
    return sampled.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, StandardScaler, pd.DataFrame]:
    """Clean data and engineer scaled model features."""
    validate_creditcard_schema(df)
    cleaned_df = df.copy()[required_creditcard_columns()]
    cleaned_df = cleaned_df.drop_duplicates().dropna().reset_index(drop=True)
    cleaned_df["Class"] = cleaned_df["Class"].astype(int)
    validate_creditcard_schema(cleaned_df)

    scaler = StandardScaler()
    feature_df = cleaned_df.copy()
    feature_df[["Amount_Scaled", "Time_Scaled"]] = scaler.fit_transform(
        feature_df[["Amount", "Time"]]
    )
    feature_df = feature_df.drop(columns=["Amount", "Time"])
    X = feature_df.drop(columns=["Class"])
    y = feature_df["Class"]
    return X, y, scaler, cleaned_df


def apply_smote_safely(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Balance the training data using SMOTE, with RandomOverSampler fallback."""
    minority_count = int(y_train.value_counts().min())
    if minority_count >= 6:
        sampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=min(5, minority_count - 1))
    else:
        sampler = RandomOverSampler(random_state=RANDOM_STATE)
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled, name="Class")


def get_positive_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Get fraud-class probabilities from a trained model."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))
    raise AttributeError("Model must support predict_proba or decision_function.")


def calculate_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> Dict[str, float]:
    """Calculate classification metrics at a threshold."""
    y_pred = (probabilities >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, probabilities),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def build_prediction_dataframe(
    model: Any,
    X: pd.DataFrame,
    raw_df: Optional[pd.DataFrame] = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Build a prediction results DataFrame."""
    probabilities = get_positive_scores(model, X)
    results = raw_df.reset_index(drop=True).copy() if raw_df is not None else X.reset_index(drop=True).copy()
    results["fraud_probability"] = probabilities
    results["is_fraud"] = results["fraud_probability"] >= threshold
    results["risk_level"] = results["fraud_probability"].apply(get_risk_level)
    return results


def save_training_artifacts(artifacts: Dict[str, Any]) -> bool:
    """Save model, scaler, and supporting artifacts if filesystem allows it."""
    try:
        ensure_directories()
        joblib.dump(artifacts["model"], MODEL_PATH)
        joblib.dump(artifacts["scaler"], SCALER_PATH)
        lightweight = {
            "metrics": artifacts["metrics"],
            "feature_names": artifacts["feature_names"],
            "X_test": artifacts["X_test"],
            "y_test": artifacts["y_test"],
            "test_raw": artifacts["test_raw"],
            "shap_sample": artifacts.get("shap_sample"),
            "shap_values": artifacts.get("shap_values"),
            "source": artifacts.get("source", "unknown"),
        }
        joblib.dump(lightweight, TRAINING_ARTIFACT_PATH)
        joblib.dump(artifacts["X_test"], PROCESSED_DIR / "X_test.pkl")
        joblib.dump(artifacts["y_test"], PROCESSED_DIR / "y_test.pkl")
        return True
    except Exception as exc:
        logger.warning("Could not save training artifacts: %s", exc)
        return False


@st.cache_resource(show_spinner=False)
def load_disk_artifacts_cached(model_mtime: float, scaler_mtime: float) -> Optional[Dict[str, Any]]:
    """Load trained model artifacts from disk and cache them."""
    del model_mtime, scaler_mtime
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        return None
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        artifacts: Dict[str, Any] = {
            "model": model,
            "scaler": scaler,
            "feature_names": model_feature_columns(),
            "source": "disk",
            "saved_to_disk": True,
        }
        if TRAINING_ARTIFACT_PATH.exists():
            artifacts.update(joblib.load(TRAINING_ARTIFACT_PATH))
        elif (PROCESSED_DIR / "X_test.pkl").exists() and (PROCESSED_DIR / "y_test.pkl").exists():
            X_test = joblib.load(PROCESSED_DIR / "X_test.pkl")
            y_test = joblib.load(PROCESSED_DIR / "y_test.pkl")
            if not isinstance(X_test, pd.DataFrame):
                X_test = pd.DataFrame(X_test, columns=model_feature_columns())
            if not isinstance(y_test, pd.Series):
                y_test = pd.Series(y_test, name="Class")
            probabilities = get_positive_scores(model, X_test)
            artifacts.update(
                {
                    "X_test": X_test,
                    "y_test": y_test,
                    "test_raw": None,
                    "metrics": calculate_metrics(y_test, probabilities, DEFAULT_THRESHOLD),
                }
            )
        return artifacts
    except Exception as exc:
        logger.warning("Could not load disk artifacts: %s", exc)
        return None


@st.cache_resource(show_spinner=False)
def train_model_cached(source_kind: str, uploaded_bytes: Optional[bytes], demo_seed: int) -> Dict[str, Any]:
    """Train the PayGuard model inside Streamlit and cache the result."""
    if source_kind == "demo":
        raw_df = generate_synthetic_creditcard_data(n_rows=DEMO_ROWS, seed=demo_seed)
        source_label = "demo"
    else:
        if uploaded_bytes is None:
            raise ValueError("No uploaded file bytes were provided.")
        from io import BytesIO

        raw_df = pd.read_csv(BytesIO(uploaded_bytes))
        source_label = "uploaded_csv"

    raw_df = maybe_sample_large_dataset(raw_df)
    X, y, scaler, cleaned_raw = prepare_features(raw_df)
    X_train, X_test, y_train, y_test, _, raw_test = train_test_split(
        X,
        y,
        cleaned_raw,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train_resampled, y_train_resampled = apply_smote_safely(X_train, y_train)

    model = XGBClassifier(
        n_estimators=160,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.90,
        colsample_bytree=0.90,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_resampled, y_train_resampled)

    probabilities = get_positive_scores(model, X_test)
    metrics = calculate_metrics(y_test, probabilities, DEFAULT_THRESHOLD)

    shap_values = None
    shap_sample = None
    if SHAP_AVAILABLE:
        try:
            shap_sample = X_test.sample(n=min(400, len(X_test)), random_state=RANDOM_STATE)
            explainer = shap.TreeExplainer(model)
            shap_values = normalise_shap_values(explainer.shap_values(shap_sample))
        except Exception as exc:
            logger.warning("SHAP generation failed: %s", exc)

    artifacts: Dict[str, Any] = {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "feature_names": X.columns.tolist(),
        "X_test": X_test.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "test_raw": raw_test.reset_index(drop=True),
        "shap_sample": shap_sample,
        "shap_values": shap_values,
        "source": source_label,
    }
    artifacts["saved_to_disk"] = save_training_artifacts(artifacts)
    return artifacts


def reset_training_state() -> None:
    """Reset session training state and force setup page."""
    for key in [
        "trained_artifacts",
        "analysis_results",
        "uploaded_analysis_df",
        "analysis_X",
        "setup_source_kind",
        "setup_uploaded_bytes",
        "setup_uploaded_name",
        "training_error",
    ]:
        st.session_state.pop(key, None)
    st.session_state["force_setup"] = True
    train_model_cached.clear()
    st.rerun()


def get_active_artifacts() -> Optional[Dict[str, Any]]:
    """Return session or disk model artifacts."""
    if "trained_artifacts" in st.session_state:
        return st.session_state["trained_artifacts"]
    if st.session_state.get("force_setup", False):
        return None
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        return load_disk_artifacts_cached(MODEL_PATH.stat().st_mtime, SCALER_PATH.stat().st_mtime)
    return None


def preprocess_new_transactions(df: pd.DataFrame, scaler: StandardScaler) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess uploaded transactions for prediction."""
    if df.empty:
        raise ValueError("Uploaded file is empty.")
    input_df = df.copy()
    display_df = input_df.copy()
    if "Class" in input_df.columns:
        input_df = input_df.drop(columns=["Class"])

    has_raw = {"Time", "Amount"}.issubset(input_df.columns)
    has_scaled = {"Time_Scaled", "Amount_Scaled"}.issubset(input_df.columns)
    if has_raw:
        missing_v = [f"V{i}" for i in range(1, 29) if f"V{i}" not in input_df.columns]
        if missing_v:
            raise ValueError(f"Missing V feature columns: {missing_v}")
        input_df[["Amount_Scaled", "Time_Scaled"]] = scaler.transform(input_df[["Amount", "Time"]])
        input_df = input_df.drop(columns=["Amount", "Time"])
    elif not has_scaled:
        raise ValueError(
            "CSV must contain either raw Time and Amount columns or scaled Time_Scaled and Amount_Scaled columns."
        )

    missing_features = [feature for feature in model_feature_columns() if feature not in input_df.columns]
    if missing_features:
        raise ValueError(f"Missing model features: {missing_features}")
    return input_df[model_feature_columns()], display_df


def get_top_factors_from_importance(model: Any, X_instance: pd.DataFrame, top_n: int = 5) -> List[Dict[str, Any]]:
    """Get top risk factors using model importances and feature values."""
    try:
        importances = getattr(model, "feature_importances_", np.ones(X_instance.shape[1]) / X_instance.shape[1])
        values = X_instance.iloc[0].to_numpy(dtype=float)
        factors_df = pd.DataFrame(
            {
                "feature": X_instance.columns,
                "value": values,
                "importance": importances,
                "impact_score": np.abs(values) * importances,
            }
        )
        return factors_df.sort_values("impact_score", ascending=False).head(top_n).to_dict(orient="records")
    except Exception:
        return []


def explain_feature_plain_english(feature: str) -> str:
    """Convert feature names into plain-English explanations."""
    explanations = {
        "Amount_Scaled": "The transaction amount is unusual compared with normal payment values.",
        "Time_Scaled": "The timing pattern differs from typical transaction activity.",
        "V14": "V14 relates to a transaction signature pattern strongly linked to fraud risk.",
        "V17": "V17 captures hidden behavioural differences often seen in fraud-like payments.",
        "V12": "V12 is another strong fraud indicator from the anonymized transaction profile.",
    }
    if feature in explanations:
        return explanations[feature]
    if feature.startswith("V"):
        return f"{feature} is an anonymized transaction signature feature that contributed to the fraud score."
    return "This factor influenced the model's decision."


def show_setup_page() -> None:
    """Render setup page when no model is available."""
    show_setup_header()
    st.markdown(
        """
        <div class="warning-banner">
            No trained model was found. Train one now by uploading the Kaggle
            creditcard.csv file or start instantly with Demo Mode.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_upload, col_demo = st.columns(2)
    with col_upload:
        st.markdown("### 📤 Upload creditcard.csv")
        st.markdown("Use columns: `Time`, `V1`-`V28`, `Amount`, and `Class`.")
        uploaded_file = st.file_uploader(
            "Upload creditcard.csv",
            type=["csv"],
            help="Expected schema: Time, V1-V28, Amount, Class.",
        )
        if uploaded_file is not None:
            st.session_state["setup_source_kind"] = "upload"
            st.session_state["setup_uploaded_bytes"] = uploaded_file.getvalue()
            st.session_state["setup_uploaded_name"] = uploaded_file.name
            st.success(f"Uploaded `{uploaded_file.name}` successfully.")

    with col_demo:
        st.markdown("### ⚡ Demo Mode")
        st.markdown("Generate a built-in synthetic dataset with 5,000 rows for instant testing.")
        if st.button("Use Demo Mode", type="primary", use_container_width=True):
            st.session_state["setup_source_kind"] = "demo"
            st.session_state["setup_uploaded_bytes"] = None
            st.session_state["setup_uploaded_name"] = "Synthetic demo dataset"
            st.success("Demo Mode selected. Ready to train.")

    source_kind = st.session_state.get("setup_source_kind")
    source_name = st.session_state.get("setup_uploaded_name")
    if source_kind:
        st.divider()
        st.markdown("### Ready to train")
        st.info(f"Selected source: **{source_name}**")
        if st.button("Train Model", type="primary", use_container_width=True):
            st.session_state.pop("training_error", None)
            run_training_progress(source_kind, st.session_state.get("setup_uploaded_bytes"))

    if "training_error" in st.session_state:
        st.error(st.session_state["training_error"])
        if st.button("Retry Setup"):
            st.session_state.pop("training_error", None)
            st.rerun()


def run_training_progress(source_kind: str, uploaded_bytes: Optional[bytes]) -> None:
    """Run model training with full-screen Streamlit progress UI."""
    st.session_state["force_setup"] = False
    st.markdown("## Training PayGuard Model")
    progress_bar = st.progress(0)

    try:
        with st.status("⚙️ Loading and validating data...", expanded=True) as status:
            time.sleep(2)
            progress_bar.progress(10)
            status.update(label="✅ Data source selected", state="complete")

        with st.status("🧹 Cleaning and scaling features...", expanded=True) as status:
            time.sleep(2)
            progress_bar.progress(25)
            status.update(label="✅ Cleaning and scaling prepared", state="complete")

        with st.status("⚖️ Balancing classes with SMOTE...", expanded=True) as status:
            time.sleep(5)
            progress_bar.progress(45)
            status.update(label="✅ Class balancing configured", state="complete")

        with st.status("🤖 Training XGBoost model...", expanded=True) as status:
            with st.spinner("Training XGBoost and evaluating fraud detection performance..."):
                artifacts = train_model_cached(source_kind, uploaded_bytes, RANDOM_STATE)
            progress_bar.progress(80)
            status.update(label="✅ XGBoost model trained", state="complete")

        with st.status("📊 Generating SHAP values...", expanded=True) as status:
            time.sleep(5)
            progress_bar.progress(95)
            if artifacts.get("shap_values") is not None:
                status.update(label="✅ SHAP values generated", state="complete")
            else:
                status.update(label="⚠️ SHAP skipped or unavailable", state="complete")

        with st.status("✅ Model ready!", expanded=True) as status:
            time.sleep(1)
            progress_bar.progress(100)
            status.update(label="✅ Model ready!", state="complete")

        st.session_state["trained_artifacts"] = artifacts
        metrics = artifacts["metrics"]
        success_banner(f"✅ Model trained successfully — AUC: {metrics['auc']:.3f} | Ready to detect fraud")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_card("Model AUC", f"{metrics['auc']:.3f}")
        with col2:
            metric_card("Precision", f"{metrics['precision']:.3f}")
        with col3:
            metric_card("Recall", f"{metrics['recall']:.3f}")
        with col4:
            metric_card("F1", f"{metrics['f1']:.3f}")

        if artifacts.get("saved_to_disk"):
            st.success("Model saved to `models/fraud_model.pkl`.")
        else:
            st.info("Model is cached in memory for this session.")

        if st.button("Open Dashboard", type="primary", use_container_width=True):
            st.rerun()

    except Exception as exc:
        logger.exception("Training failed.")
        st.session_state["training_error"] = f"Training failed. Please try again. Details: {exc}"
        st.error(st.session_state["training_error"])
        if st.button("Retry Training"):
            st.session_state.pop("training_error", None)
            st.rerun()


def page_overview(artifacts: Dict[str, Any]) -> None:
    """Render Overview page."""
    show_header("AI-Powered Payment Fraud Detection", "real-time risk monitoring for transaction teams")
    model = artifacts["model"]
    X_test = artifacts.get("X_test")
    y_test = artifacts.get("y_test")
    test_raw = artifacts.get("test_raw")

    if X_test is None or y_test is None:
        demo_raw = generate_synthetic_creditcard_data(1_000, seed=99)
        X_test, y_test, _, test_raw = prepare_features(demo_raw)

    results_df = build_prediction_dataframe(model, X_test, test_raw)
    if "Time" in results_df.columns:
        results_df["transaction_time"] = pd.to_datetime("2026-01-01") + pd.to_timedelta(
            results_df["Time"], unit="s"
        )
    else:
        results_df["transaction_time"] = pd.date_range("2026-01-01", periods=len(results_df), freq="h")

    total_transactions = len(results_df)
    fraud_detected = int(results_df["is_fraud"].sum())
    high_risk_df = results_df[results_df["risk_level"].isin(["HIGH", "CRITICAL"])]
    amount_protected = float(high_risk_df["Amount"].sum()) if "Amount" in high_risk_df.columns else 0.0
    accuracy_text = "N/A"
    if y_test is not None:
        accuracy = artifacts.get("metrics", {}).get(
            "accuracy", accuracy_score(y_test, results_df["is_fraud"].astype(int))
        )
        accuracy_text = f"{accuracy * 100:.2f}%"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Total Transactions Analysed", f"{total_transactions:,}")
    with col2:
        metric_card("Fraud Detected", f"{fraud_detected:,}")
    with col3:
        metric_card("Amount Protected", format_currency(amount_protected))
    with col4:
        metric_card("Model Accuracy", accuracy_text)

    st.markdown("### Fraud Monitoring Trends")
    chart_col1, chart_col2 = st.columns([2, 1])
    with chart_col1:
        trend_df = (
            results_df.assign(fraud_flag=results_df["is_fraud"].astype(int))
            .set_index("transaction_time")
            .resample("6h")["fraud_flag"]
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
            results_df["risk_level"]
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


def page_analyse_transactions(artifacts: Dict[str, Any]) -> None:
    """Render Analyse Transactions page."""
    show_header("Analyse Transactions", "upload a CSV and run fraud scoring")
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"],
        help="CSV should include V1-V28 and either Time/Amount or Time_Scaled/Amount_Scaled.",
    )
    if uploaded_file is None:
        st.info("Upload a transaction CSV to begin analysis.")
        return

    try:
        uploaded_df = pd.read_csv(uploaded_file)
        st.session_state["uploaded_analysis_df"] = uploaded_df
        st.markdown("### Uploaded Data Preview")
        st.dataframe(uploaded_df.head(), use_container_width=True)

        if st.button("Run Fraud Analysis", type="primary"):
            with st.spinner("Running fraud analysis..."):
                X_input, display_df = preprocess_new_transactions(uploaded_df, scaler)
                results_df = build_prediction_dataframe(model, X_input, display_df)
                st.session_state["analysis_results"] = results_df
                st.session_state["analysis_X"] = X_input

        if "analysis_results" in st.session_state:
            results_df = st.session_state["analysis_results"]
            st.markdown("### Fraud Analysis Results")
            st.dataframe(styled_risk_table(results_df.head(100)), use_container_width=True)
            total_analysed = len(results_df)
            high_risk_mask = results_df["risk_level"].isin(["HIGH", "CRITICAL"])
            high_risk_count = int(high_risk_mask.sum())
            amount_at_risk = float(results_df.loc[high_risk_mask, "Amount"].sum()) if "Amount" in results_df.columns else 0.0

            col1, col2, col3 = st.columns(3)
            with col1:
                metric_card("Transactions Analysed", f"{total_analysed:,}")
            with col2:
                metric_card("High-Risk Flagged", f"{high_risk_count:,}")
            with col3:
                metric_card("Amount at Risk", format_currency(amount_at_risk))

            st.download_button(
                label="Download Results as CSV",
                data=results_df.to_csv(index=False).encode("utf-8"),
                file_name="payguard_fraud_analysis_results.csv",
                mime="text/csv",
            )
    except Exception as exc:
        logger.exception("Analysis failed.")
        st.error(f"Could not analyse uploaded file: {exc}")


def page_model_explainability(artifacts: Dict[str, Any]) -> None:
    """Render Model Explainability page."""
    show_header("Model Explainability", "understand why a transaction was flagged")
    st.markdown("## Why did the model flag this transaction?")
    model = artifacts["model"]

    if "analysis_results" in st.session_state and "analysis_X" in st.session_state:
        results_df = st.session_state["analysis_results"]
        X_source = st.session_state["analysis_X"]
        source_label = "uploaded analysis results"
    else:
        X_source = artifacts.get("X_test")
        test_raw = artifacts.get("test_raw")
        if X_source is None:
            st.info("No analysed transactions available yet.")
            return
        results_df = build_prediction_dataframe(model, X_source, test_raw)
        source_label = "test sample"

    st.caption(f"Explaining transactions from: {source_label}")
    selected_id = st.selectbox("Select Transaction ID", options=list(range(len(results_df))))
    selected_raw = results_df.iloc[[selected_id]].copy()
    selected_X = X_source.iloc[[selected_id]].copy()
    probability = float(get_positive_scores(model, selected_X)[0])
    risk_level = get_risk_level(probability)

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Fraud Probability", f"{probability * 100:.2f}%")
    with col2:
        metric_card("Risk Level", risk_level)
    with col3:
        metric_card("Decision", "Fraud" if probability >= DEFAULT_THRESHOLD else "Legitimate")

    st.markdown("### SHAP Waterfall Chart")
    shap_top_factors: List[Dict[str, Any]] = []
    if not SHAP_AVAILABLE:
        st.warning("SHAP or Matplotlib is not installed. Showing fallback risk factors.")
    else:
        try:
            with st.spinner("Generating SHAP explanation..."):
                explainer = shap.TreeExplainer(model)
                local_values = normalise_shap_values(explainer.shap_values(selected_X))[0]
                expected_value = explainer.expected_value
                if isinstance(expected_value, (list, np.ndarray)):
                    expected_value = float(np.asarray(expected_value).ravel()[-1])
                explanation = shap.Explanation(
                    values=local_values,
                    base_values=float(expected_value),
                    data=selected_X.iloc[0].values,
                    feature_names=selected_X.columns.tolist(),
                )
                fig = plt.figure(figsize=(10, 6))
                shap.plots.waterfall(explanation, max_display=10, show=False)
                st.pyplot(fig, clear_figure=True)
                shap_df = pd.DataFrame(
                    {
                        "feature": selected_X.columns,
                        "value": selected_X.iloc[0].values,
                        "impact_score": np.abs(local_values),
                        "shap_value": local_values,
                    }
                )
                shap_top_factors = shap_df.sort_values("impact_score", ascending=False).head(5).to_dict(orient="records")
        except Exception as exc:
            logger.warning("SHAP local explanation failed: %s", exc)
            st.warning("SHAP explanation could not be generated. Showing fallback risk factors.")

    top_factors = shap_top_factors or get_top_factors_from_importance(model, selected_X)
    st.markdown("### Top 5 Risk Factors")
    for factor in top_factors[:5]:
        feature = str(factor.get("feature", "Unknown"))
        value = float(factor.get("value", 0.0))
        impact = float(factor.get("impact_score", 0.0))
        st.markdown(
            f"""
            <div class="risk-card">
                <div class="factor-title">{feature}</div>
                <div class="factor-body">
                    Value: <b>{value:.4f}</b> | Impact score: <b>{impact:.4f}</b><br>
                    {explain_feature_plain_english(feature)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Selected transaction preview"):
        st.dataframe(selected_raw, use_container_width=True)


def page_model_performance(artifacts: Dict[str, Any]) -> None:
    """Render Model Performance page."""
    show_header("Model Performance", "evaluate fraud detection quality live")
    model = artifacts["model"]
    X_test = artifacts.get("X_test")
    y_test = artifacts.get("y_test")
    if X_test is None or y_test is None:
        st.warning("No labelled test data is available for performance charts.")
        return

    probabilities = get_positive_scores(model, X_test)
    threshold = st.slider(
        "Fraud Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=DEFAULT_THRESHOLD,
        step=0.05,
        help="Lower threshold catches more fraud but may increase false positives.",
    )
    metrics = calculate_metrics(y_test, probabilities, threshold)
    y_pred = (probabilities >= threshold).astype(int)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Precision", f"{metrics['precision']:.3f}")
    with col2:
        metric_card("Recall", f"{metrics['recall']:.3f}")
    with col3:
        metric_card("F1", f"{metrics['f1']:.3f}")
    with col4:
        metric_card("AUC", f"{metrics['auc']:.3f}")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        roc_auc = roc_auc_score(y_test, probabilities)
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC AUC = {roc_auc:.4f}"))
        roc_fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Classifier", line=dict(dash="dash"))
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
        pr_fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"PR AUC = {pr_auc:.4f}"))
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
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1": metrics["f1"],
                    "AUC": metrics["auc"],
                    "Accuracy": metrics["accuracy"],
                }
            ]
        )
        st.dataframe(metrics_df.round(4), use_container_width=True)
        st.markdown(
            """
            **How to read this:**
            - Higher **recall** catches more real fraud.
            - Higher **precision** means fewer false fraud alerts.
            - Lower thresholds usually increase recall but reduce precision.
            """
        )


def render_sidebar(artifacts: Dict[str, Any]) -> str:
    """Render sidebar and return selected page."""
    st.sidebar.markdown("## 🛡️ PayGuard")
    st.sidebar.markdown("Self-contained fraud detection dashboard")
    metrics = artifacts.get("metrics", {})
    if metrics:
        st.sidebar.success(f"Model ready | AUC {metrics.get('auc', 0):.3f}")
    else:
        st.sidebar.success("Model ready")
    st.sidebar.caption("Model source: disk/session cache" if artifacts.get("saved_to_disk") else "Model source: in-memory session")
    if st.sidebar.button("Retrain Model", use_container_width=True):
        reset_training_state()
    st.sidebar.markdown("---")
    return st.sidebar.radio(
        "Navigation",
        options=[
            "🏠 Overview",
            "🔍 Analyse Transactions",
            "🧠 Model Explainability",
            "📊 Model Performance",
        ],
    )


def main() -> None:
    """Run the PayGuard Streamlit app."""
    ensure_directories()
    artifacts = get_active_artifacts()
    if artifacts is None:
        show_setup_page()
        return

    metrics = artifacts.get("metrics")
    if metrics:
        success_banner(f"✅ Model trained successfully — AUC: {metrics['auc']:.3f} | Ready to detect fraud")

    selected_page = render_sidebar(artifacts)
    try:
        if selected_page == "🏠 Overview":
            page_overview(artifacts)
        elif selected_page == "🔍 Analyse Transactions":
            page_analyse_transactions(artifacts)
        elif selected_page == "🧠 Model Explainability":
            page_model_explainability(artifacts)
        elif selected_page == "📊 Model Performance":
            page_model_performance(artifacts)
        else:
            st.error("Unknown page selected.")
    except Exception as exc:
        logger.exception("Dashboard error.")
        st.error(f"Something went wrong: {exc}")
        if st.button("Return to Setup"):
            reset_training_state()


if __name__ == "__main__":
    main()

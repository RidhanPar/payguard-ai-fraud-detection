"""
Additional coverage tests for PayGuard.

These tests focus on utility functions, SHAP explainability, pipeline persistence,
model comparison, threshold metrics, and error-handling paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
import pytest

from src.data_pipeline import FraudDataPipeline
from src.explain import FraudExplainer
from src.predict import FraudPredictor
from src.train import ModelTrainer
from src.utils import calculate_business_impact, format_currency, load_sample_data


def test_run_pipeline_saves_processed_files(pipeline_config: Dict[str, object]) -> None:
    """
    Test that the full data pipeline runs end-to-end and saves processed files.

    This validates the production preprocessing path used before model training.
    """
    pipeline = FraudDataPipeline(config=pipeline_config)

    X_train, X_test, y_train, y_test = pipeline.run_pipeline()

    processed_dir = Path(str(pipeline_config["PROCESSED_DIR"]))
    model_dir = Path(str(pipeline_config["MODEL_DIR"]))

    assert not X_train.empty
    assert not X_test.empty
    assert not y_train.empty
    assert not y_test.empty
    assert (processed_dir / "X_train.pkl").exists()
    assert (processed_dir / "X_test.pkl").exists()
    assert (processed_dir / "y_train.pkl").exists()
    assert (processed_dir / "y_test.pkl").exists()
    assert (model_dir / "scaler.pkl").exists()


def test_compare_models_returns_sorted_dataframe(
    trained_model,
    processed_dataset: Dict[str, object],
) -> None:
    """
    Test that compare_models returns a model comparison DataFrame.

    This supports the modelling notebook and dashboard performance comparison.
    """
    trainer = ModelTrainer(random_state=42)

    metrics = trainer.evaluate_model(
        trained_model,
        processed_dataset["X"],
        processed_dataset["y"],
    )

    comparison_df = trainer.compare_models({"Random Forest": metrics})

    assert isinstance(comparison_df, pd.DataFrame)
    assert "model" in comparison_df.columns
    assert comparison_df.loc[0, "model"] == "Random Forest"


def test_tune_xgboost_returns_fitted_grid_search(
    processed_dataset: Dict[str, object],
) -> None:
    """
    Test that XGBoost hyperparameter tuning returns a fitted GridSearchCV object.

    This verifies the final model optimization path used by PayGuard.
    """
    trainer = ModelTrainer(random_state=42)

    grid_search = trainer.tune_xgboost(
        processed_dataset["X"],
        processed_dataset["y"],
    )

    assert hasattr(grid_search, "best_estimator_")
    assert hasattr(grid_search, "best_params_")
    assert grid_search.best_estimator_ is not None


def test_get_threshold_metrics_returns_required_values(
    predictor: FraudPredictor,
    sample_dataframe: pd.DataFrame,
) -> None:
    """
    Test that custom-threshold metrics return precision, recall, and F1.

    This is used by the Streamlit dashboard threshold slider.
    """
    batch_df = sample_dataframe.head(20).copy()
    prediction_df = predictor.predict_batch(batch_df.drop(columns=["Class"]))
    prediction_df["Class"] = batch_df["Class"].values

    metrics = predictor.get_threshold_metrics(prediction_df, threshold=0.5)

    assert {"threshold", "precision", "recall", "f1"}.issubset(metrics.keys())
    assert metrics["threshold"] == 0.5
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0


def test_get_risk_level_rejects_invalid_probability(predictor: FraudPredictor) -> None:
    """
    Test that invalid probability values are rejected.

    This protects dashboard and API usage from invalid risk-band inputs.
    """
    with pytest.raises(ValueError):
        predictor.get_risk_level(1.5)


def test_format_currency_returns_pound_string() -> None:
    """
    Test that currency formatting returns a readable pound value.

    This is used in dashboard metric cards and business-impact summaries.
    """
    assert format_currency(1234.5) == "£1,234.50"


def test_format_currency_rejects_invalid_amount() -> None:
    """
    Test that invalid currency values raise a clear ValueError.

    This prevents silent failures in business-impact reporting.
    """
    with pytest.raises(ValueError):
        format_currency("not-a-number")


def test_calculate_business_impact_returns_total_saved() -> None:
    """
    Test that business impact returns the protected fraud amount.

    This keeps the dashboard calculation simple and explainable.
    """
    assert calculate_business_impact(fraud_caught=5, total_fraud_amount=2500.75) == 2500.75


def test_calculate_business_impact_rejects_negative_values() -> None:
    """
    Test that negative fraud counts or amounts are rejected.

    Business-impact values should never be negative.
    """
    with pytest.raises(ValueError):
        calculate_business_impact(fraud_caught=-1, total_fraud_amount=100.0)


def test_load_sample_data_returns_requested_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    processed_dataset: Dict[str, object],
) -> None:
    """
    Test that load_sample_data returns rows from processed test files.

    This supports sample transaction loading in the dashboard.
    """
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)

    X = processed_dataset["X"]
    y = processed_dataset["y"]

    joblib.dump(X, processed_dir / "X_test.pkl")
    joblib.dump(y, processed_dir / "y_test.pkl")

    monkeypatch.chdir(tmp_path)

    sample = load_sample_data(5)

    assert isinstance(sample, pd.DataFrame)
    assert len(sample) == 5
    assert "Class" in sample.columns


def test_fraud_explainer_returns_feature_importance(
    trained_model,
    processed_dataset: Dict[str, object],
) -> None:
    """
    Test that FraudExplainer returns sorted SHAP feature importance.

    This verifies the global explainability layer used by PayGuard.
    """
    X = processed_dataset["X"].head(30)

    explainer = FraudExplainer(model=trained_model, X_train=X)
    importance_df = explainer.get_feature_importance_df()

    assert isinstance(importance_df, pd.DataFrame)
    assert {"feature", "shap_importance"}.issubset(importance_df.columns)
    assert not importance_df.empty


def test_fraud_explainer_explain_prediction_returns_top_factors(
    trained_model,
    processed_dataset: Dict[str, object],
) -> None:
    """
    Test that FraudExplainer explains one transaction with top factors.

    This supports local explainability for individual fraud decisions.
    """
    X = processed_dataset["X"].head(30)
    X_instance = X.head(1)

    explainer = FraudExplainer(model=trained_model, X_train=X)
    explanation = explainer.explain_prediction(X_instance)

    assert {
        "prediction",
        "fraud_probability",
        "base_value",
        "model_output_value",
        "top_5_factors",
    }.issubset(explanation.keys())
    assert isinstance(explanation["top_5_factors"], list)
    assert len(explanation["top_5_factors"]) <= 5


def test_fraud_explainer_plot_summary_saves_file(
    tmp_path: Path,
    trained_model,
    processed_dataset: Dict[str, object],
) -> None:
    """
    Test that FraudExplainer can save a SHAP summary plot.

    This verifies that explainability visuals can be generated as report assets.
    """
    X = processed_dataset["X"].head(30)

    explainer = FraudExplainer(model=trained_model, X_train=X)
    shap_values = explainer.get_shap_values(X)

    output_path = tmp_path / "figures" / "summary.png"
    saved_path = explainer.plot_summary(
        shap_values=shap_values,
        X=X,
        plot_type="bar",
        output_path=output_path,
    )

    assert saved_path.exists()
    assert saved_path.suffix == ".png"


def test_fraud_explainer_rejects_empty_training_data(
    trained_model,
    processed_dataset: Dict[str, object],
) -> None:
    """
    Test that FraudExplainer rejects empty training data.

    This protects explainability setup from invalid inputs.
    """
    empty_X = processed_dataset["X"].iloc[0:0]

    with pytest.raises(ValueError):
        FraudExplainer(model=trained_model, X_train=empty_X)

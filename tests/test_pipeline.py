"""
Unit tests for the PayGuard data pipeline, model trainer, and predictor.

Run from the project root:

    pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import FraudDataPipeline
from src.predict import FraudPredictor
from src.train import ModelTrainer

# ---------------------------------------------------------------------------
# 1. TEST DATA PIPELINE
# ---------------------------------------------------------------------------


def test_load_data_returns_dataframe(pipeline_config: Dict[str, object]) -> None:
    """
    Test that FraudDataPipeline.load_data returns a pandas DataFrame.

    This confirms the pipeline can read the configured raw CSV file correctly.
    """
    pipeline = FraudDataPipeline(config=pipeline_config)

    df = pipeline.load_data()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_data_has_correct_columns(pipeline_config: Dict[str, object]) -> None:
    """
    Test that loaded data contains all required PayGuard columns.

    The model pipeline depends on Time, Amount, Class, and V1-V28.
    """
    pipeline = FraudDataPipeline(config=pipeline_config)

    df = pipeline.load_data()

    expected_columns = {"Time", "Amount", "Class", *[f"V{i}" for i in range(1, 29)]}

    assert expected_columns.issubset(set(df.columns))


def test_no_null_values_after_cleaning(
    pipeline_config: Dict[str, object],
    sample_dataframe: pd.DataFrame,
) -> None:
    """
    Test that clean_data removes rows with null values.

    This protects downstream model training from missing-value errors.
    """
    df_with_null = sample_dataframe.copy()
    df_with_null.loc[0, "Amount"] = np.nan

    pipeline = FraudDataPipeline(config=pipeline_config)
    cleaned_df = pipeline.clean_data(df_with_null)

    assert cleaned_df.isnull().sum().sum() == 0


def test_class_distribution_after_smote(pipeline_config: Dict[str, object]) -> None:
    """
    Test that SMOTE balances the training class distribution.

    Fraud detection data is imbalanced, so the training set should contain
    equal class counts after SMOTE resampling.
    """
    pipeline = FraudDataPipeline(config=pipeline_config)

    df = pipeline.load_data()
    cleaned_df = pipeline.clean_data(df)
    engineered_df = pipeline.engineer_features(cleaned_df)
    X_train, _, y_train, _ = pipeline.split_data(engineered_df)

    X_resampled, y_resampled = pipeline.apply_smote(X_train, y_train)

    class_counts = pd.Series(y_resampled).value_counts()

    assert X_resampled.shape[0] == y_resampled.shape[0]
    assert class_counts.loc[0] == class_counts.loc[1]


def test_feature_scaling_range(pipeline_config: Dict[str, object]) -> None:
    """
    Test that Amount and Time are standard-scaled correctly.

    StandardScaler should transform Amount_Scaled and Time_Scaled to have
    approximately zero mean and unit variance.
    """
    pipeline = FraudDataPipeline(config=pipeline_config)

    df = pipeline.load_data()
    cleaned_df = pipeline.clean_data(df)
    engineered_df = pipeline.engineer_features(cleaned_df)

    assert "Amount" not in engineered_df.columns
    assert "Time" not in engineered_df.columns
    assert "Amount_Scaled" in engineered_df.columns
    assert "Time_Scaled" in engineered_df.columns

    assert np.isclose(engineered_df["Amount_Scaled"].mean(), 0.0, atol=1e-8)
    assert np.isclose(engineered_df["Time_Scaled"].mean(), 0.0, atol=1e-8)
    assert np.isclose(engineered_df["Amount_Scaled"].std(ddof=0), 1.0, atol=1e-8)
    assert np.isclose(engineered_df["Time_Scaled"].std(ddof=0), 1.0, atol=1e-8)


# ---------------------------------------------------------------------------
# 2. TEST MODEL
# ---------------------------------------------------------------------------


def test_logistic_regression_trains_without_error(
    processed_dataset: Dict[str, object],
) -> None:
    """
    Test that Logistic Regression trains without raising an error.

    This verifies the baseline model can fit the processed training schema.
    """
    trainer = ModelTrainer(random_state=42)

    model = trainer.train_logistic_regression(
        processed_dataset["X"],
        processed_dataset["y"],
    )

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_random_forest_trains_without_error(
    processed_dataset: Dict[str, object],
) -> None:
    """
    Test that Random Forest trains without raising an error.

    This confirms the tree-based model works with the PayGuard feature matrix.
    """
    trainer = ModelTrainer(random_state=42)

    model = trainer.train_random_forest(
        processed_dataset["X"],
        processed_dataset["y"],
    )

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_xgboost_trains_without_error(
    processed_dataset: Dict[str, object],
) -> None:
    """
    Test that XGBoost trains without raising an error.

    XGBoost is the final selected PayGuard model, so this verifies the core
    production training path.
    """
    trainer = ModelTrainer(random_state=42)

    model = trainer.train_xgboost(
        processed_dataset["X"],
        processed_dataset["y"],
    )

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_model_saves_and_loads_correctly(
    tmp_path: Path,
    trained_model,
    processed_dataset: Dict[str, object],
) -> None:
    """
    Test that ModelTrainer can save and reload a trained model.

    This ensures model persistence works for dashboard and prediction usage.
    """
    trainer = ModelTrainer(random_state=42)
    model_path = tmp_path / "fraud_model.pkl"

    trainer.save_model(trained_model, model_path)
    loaded_model = trainer.load_model(model_path)

    original_predictions = trained_model.predict(processed_dataset["X"])
    loaded_predictions = loaded_model.predict(processed_dataset["X"])

    assert model_path.exists()
    assert np.array_equal(original_predictions, loaded_predictions)


def test_evaluate_model_returns_required_metrics(
    trained_model,
    processed_dataset: Dict[str, object],
) -> None:
    """
    Test that evaluate_model returns all required fraud detection metrics.

    These metrics are used in model comparison and dashboard performance pages.
    """
    trainer = ModelTrainer(random_state=42)

    metrics = trainer.evaluate_model(
        trained_model,
        processed_dataset["X"],
        processed_dataset["y"],
    )

    required_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
    }

    assert required_metrics.issubset(metrics.keys())

    for metric_name in required_metrics:
        assert 0.0 <= metrics[metric_name] <= 1.0


# ---------------------------------------------------------------------------
# 3. TEST PREDICTOR
# ---------------------------------------------------------------------------


def test_predict_single_returns_required_keys(
    predictor: FraudPredictor,
    mock_transaction: Dict[str, float],
) -> None:
    """
    Test that predict_single returns the expected response structure.

    The Streamlit dashboard depends on these keys for displaying prediction
    results and risk factors.
    """
    result = predictor.predict_single(mock_transaction)

    required_keys = {
        "fraud_probability",
        "is_fraud",
        "risk_level",
        "top_factors",
    }

    assert required_keys.issubset(result.keys())
    assert isinstance(result["fraud_probability"], float)
    assert isinstance(result["is_fraud"], bool)
    assert isinstance(result["risk_level"], str)
    assert isinstance(result["top_factors"], list)


@pytest.mark.parametrize(
    ("probability", "expected_risk"),
    [
        (0.00, "LOW"),
        (0.29, "LOW"),
    ],
)
def test_risk_level_low_for_zero_probability(
    predictor: FraudPredictor,
    probability: float,
    expected_risk: str,
) -> None:
    """
    Test that very low fraud probabilities are assigned LOW risk.

    This verifies the lower risk-band boundary used by the dashboard.
    """
    assert predictor.get_risk_level(probability) == expected_risk


@pytest.mark.parametrize(
    ("probability", "expected_risk"),
    [
        (0.85, "CRITICAL"),
        (0.99, "CRITICAL"),
    ],
)
def test_risk_level_critical_for_high_probability(
    predictor: FraudPredictor,
    probability: float,
    expected_risk: str,
) -> None:
    """
    Test that high fraud probabilities are assigned CRITICAL risk.

    This verifies the high-risk boundary for urgent fraud alerts.
    """
    assert predictor.get_risk_level(probability) == expected_risk


def test_predict_batch_adds_correct_columns(
    predictor: FraudPredictor,
    sample_dataframe: pd.DataFrame,
) -> None:
    """
    Test that predict_batch adds prediction output columns.

    The results table in the Streamlit dashboard requires these columns.
    """
    batch_df = sample_dataframe.drop(columns=["Class"]).head(10)

    results_df = predictor.predict_batch(batch_df)

    expected_columns = {
        "fraud_probability",
        "is_fraud",
        "risk_level",
    }

    assert expected_columns.issubset(results_df.columns)
    assert len(results_df) == len(batch_df)


def test_preprocess_input_scales_correctly(
    predictor: FraudPredictor,
    sample_dataframe: pd.DataFrame,
) -> None:
    """
    Test that preprocess_input creates scaled features and drops raw columns.

    This ensures new dashboard inputs follow the same training-time
    preprocessing logic.
    """
    raw_input = sample_dataframe.drop(columns=["Class"]).head(5)

    processed_input = predictor.preprocess_input(raw_input)

    assert "Amount" not in processed_input.columns
    assert "Time" not in processed_input.columns
    assert "Amount_Scaled" in processed_input.columns
    assert "Time_Scaled" in processed_input.columns
    assert processed_input.shape[0] == 5
    assert all(feature in processed_input.columns for feature in predictor.feature_names)

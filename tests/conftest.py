"""
Shared pytest fixtures for the PayGuard test suite.

The fixtures use a small synthetic fraud dataset so tests can run quickly
without requiring the full Kaggle creditcard.csv file.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """
    Create a 100-row synthetic PayGuard-style dataset.

    The dataset follows the Kaggle credit card fraud schema:
    Time, V1-V28, Amount, and Class.

    Returns:
        Synthetic transaction DataFrame with both legitimate and fraud rows.
    """
    rng = np.random.default_rng(42)
    n_rows = 100
    n_fraud = 10
    n_legitimate = n_rows - n_fraud

    legitimate_features = rng.normal(loc=0.0, scale=1.0, size=(n_legitimate, 28))
    fraud_features = rng.normal(loc=1.8, scale=1.2, size=(n_fraud, 28))

    feature_matrix = np.vstack([legitimate_features, fraud_features])

    df = pd.DataFrame(
        feature_matrix,
        columns=[f"V{i}" for i in range(1, 29)],
    )

    df.insert(0, "Time", rng.integers(0, 172_800, size=n_rows))
    df["Amount"] = np.concatenate(
        [
            rng.lognormal(mean=3.0, sigma=0.8, size=n_legitimate),
            rng.lognormal(mean=4.0, sigma=0.9, size=n_fraud),
        ]
    )
    df["Class"] = [0] * n_legitimate + [1] * n_fraud

    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)


@pytest.fixture
def pipeline_config(tmp_path: Path, sample_dataframe: pd.DataFrame) -> Dict[str, object]:
    """
    Create a temporary pipeline config and raw CSV file.

    Args:
        tmp_path: Pytest temporary directory.
        sample_dataframe: Synthetic PayGuard dataset.

    Returns:
        Configuration dictionary for FraudDataPipeline.
    """
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    model_dir = tmp_path / "models"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    data_path = raw_dir / "creditcard.csv"
    sample_dataframe.to_csv(data_path, index=False)

    return {
        "DATA_PATH": str(data_path),
        "PROCESSED_DIR": str(processed_dir),
        "MODEL_DIR": str(model_dir),
        "RANDOM_STATE": 42,
        "TEST_SIZE": 0.20,
    }


@pytest.fixture
def processed_dataset(sample_dataframe: pd.DataFrame) -> Dict[str, object]:
    """
    Create scaled model-ready features and labels from the sample dataset.

    Args:
        sample_dataframe: Synthetic PayGuard dataset.

    Returns:
        Dictionary containing X, y, scaler, and raw DataFrame.
    """
    scaler = StandardScaler()
    processed_df = sample_dataframe.copy()

    processed_df[["Amount_Scaled", "Time_Scaled"]] = scaler.fit_transform(
        processed_df[["Amount", "Time"]]
    )
    processed_df = processed_df.drop(columns=["Amount", "Time"])

    X = processed_df.drop(columns=["Class"])
    y = processed_df["Class"]

    return {
        "X": X,
        "y": y,
        "scaler": scaler,
        "raw_df": sample_dataframe,
    }


@pytest.fixture
def trained_model(processed_dataset: Dict[str, object]) -> RandomForestClassifier:
    """
    Train a small Random Forest model for fast prediction tests.

    Args:
        processed_dataset: Processed synthetic dataset fixture.

    Returns:
        Trained RandomForestClassifier.
    """
    X = processed_dataset["X"]
    y = processed_dataset["y"]

    model = RandomForestClassifier(
        n_estimators=20,
        max_depth=4,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    return model


@pytest.fixture
def predictor_artifacts(
    tmp_path: Path,
    trained_model: RandomForestClassifier,
    processed_dataset: Dict[str, object],
) -> Dict[str, Path]:
    """
    Save temporary model and scaler artifacts for FraudPredictor tests.

    Args:
        tmp_path: Pytest temporary directory.
        trained_model: Small trained model fixture.
        processed_dataset: Processed synthetic dataset fixture.

    Returns:
        Dictionary containing model_path and scaler_path.
    """
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "fraud_model.pkl"
    scaler_path = model_dir / "scaler.pkl"

    joblib.dump(trained_model, model_path)
    joblib.dump(processed_dataset["scaler"], scaler_path)

    return {
        "model_path": model_path,
        "scaler_path": scaler_path,
    }


@pytest.fixture
def predictor(predictor_artifacts: Dict[str, Path]):
    """
    Create a FraudPredictor instance using temporary artifacts.

    Args:
        predictor_artifacts: Saved model and scaler paths.

    Returns:
        FraudPredictor instance.
    """
    from src.predict import FraudPredictor

    return FraudPredictor(
        model_path=predictor_artifacts["model_path"],
        scaler_path=predictor_artifacts["scaler_path"],
    )


@pytest.fixture
def mock_transaction(sample_dataframe: pd.DataFrame) -> Dict[str, float]:
    """
    Create a single raw transaction dictionary for prediction tests.

    Args:
        sample_dataframe: Synthetic PayGuard dataset.

    Returns:
        Single transaction dictionary without the target column.
    """
    return sample_dataframe.drop(columns=["Class"]).iloc[0].to_dict()

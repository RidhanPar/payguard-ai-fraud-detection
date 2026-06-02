"""
Prediction utilities for the PayGuard fraud detection project.

This module provides the FraudPredictor class used by the Streamlit dashboard
and other application layers. It loads the trained fraud detection model and
the fitted scaler, preprocesses raw or already-processed transaction data, and
returns fraud risk predictions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


class FraudPredictor:
    """
    Fraud prediction service for PayGuard.

    This class loads the trained model and fitted scaler, applies the same
    preprocessing logic used during training, and returns single or batch
    fraud predictions.

    Args:
        model_path: Path to the saved trained model file.
        scaler_path: Path to the saved fitted StandardScaler file.

    Raises:
        FileNotFoundError: If the model or scaler file does not exist.
        RuntimeError: If model or scaler loading fails.
    """

    def __init__(self, model_path: str | Path, scaler_path: str | Path) -> None:
        """
        Initialize the FraudPredictor by loading model and scaler artifacts.

        Args:
            model_path: Path to the trained fraud model.
            scaler_path: Path to the fitted scaler.

        Raises:
            FileNotFoundError: If required artifact files are missing.
            RuntimeError: If loading artifacts fails.
        """
        try:
            self.model_path = Path(model_path)
            self.scaler_path = Path(scaler_path)

            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            if not self.scaler_path.exists():
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")

            self.model = joblib.load(self.model_path)
            self.scaler: StandardScaler = joblib.load(self.scaler_path)
            self.feature_names = self._get_model_feature_names()

            logger.info("FraudPredictor initialized successfully.")
            logger.info("Loaded model from %s.", self.model_path)
            logger.info("Loaded scaler from %s.", self.scaler_path)

        except FileNotFoundError:
            logger.exception("Required prediction artifact is missing.")
            raise
        except Exception as exc:
            logger.exception("Failed to initialize FraudPredictor.")
            raise RuntimeError("Failed to load prediction artifacts.") from exc

    def _get_model_feature_names(self) -> List[str]:
        """
        Get the feature names expected by the trained model.

        Returns:
            List of model feature names.

        Notes:
            XGBoost models trained with pandas usually store feature names.
            If feature names are unavailable, this method returns the expected
            PayGuard training schema.
        """
        try:
            if hasattr(self.model, "feature_names_in_"):
                return list(self.model.feature_names_in_)

            if hasattr(self.model, "get_booster"):
                booster_features = self.model.get_booster().feature_names
                if booster_features:
                    return list(booster_features)

            return [f"V{i}" for i in range(1, 29)] + [
                "Amount_Scaled",
                "Time_Scaled",
            ]

        except Exception:
            logger.warning("Could not read feature names from model. Using default schema.")
            return [f"V{i}" for i in range(1, 29)] + [
                "Amount_Scaled",
                "Time_Scaled",
            ]

    def preprocess_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the same preprocessing used in the training pipeline.

        This method supports two input formats:

        1. Raw input with `Amount` and `Time` columns.
        2. Already-processed input with `Amount_Scaled` and `Time_Scaled`.

        If raw columns are present, they are scaled using the fitted scaler and
        the original columns are dropped. The final DataFrame is aligned to the
        model's expected feature order.

        Args:
            df: Raw or processed transaction DataFrame.

        Returns:
            Model-ready feature DataFrame.

        Raises:
            ValueError: If input data is empty or required features are missing.
            RuntimeError: If preprocessing fails.
        """
        try:
            if df.empty:
                raise ValueError("Input DataFrame cannot be empty.")

            processed_df = df.copy()

            if "Class" in processed_df.columns:
                processed_df = processed_df.drop(columns=["Class"])

            has_raw_time_amount = {"Amount", "Time"}.issubset(processed_df.columns)
            has_scaled_time_amount = {
                "Amount_Scaled",
                "Time_Scaled",
            }.issubset(processed_df.columns)

            if has_raw_time_amount:
                processed_df[["Amount_Scaled", "Time_Scaled"]] = self.scaler.transform(
                    processed_df[["Amount", "Time"]]
                )
                processed_df = processed_df.drop(columns=["Amount", "Time"])

            elif not has_scaled_time_amount:
                raise ValueError(
                    "Input data must contain either raw `Amount` and `Time` columns "
                    "or processed `Amount_Scaled` and `Time_Scaled` columns."
                )

            missing_features = [
                feature for feature in self.feature_names if feature not in processed_df.columns
            ]

            if missing_features:
                raise ValueError(f"Missing model input features: {missing_features}")

            processed_df = processed_df[self.feature_names]

            logger.info("Input preprocessing completed for %s rows.", len(processed_df))
            return processed_df

        except ValueError:
            logger.exception("Input validation failed during preprocessing.")
            raise
        except Exception as exc:
            logger.exception("Failed to preprocess input data.")
            raise RuntimeError("Failed to preprocess input data.") from exc

    def _predict_probability(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud probabilities for model-ready features.

        Args:
            X: Preprocessed feature DataFrame.

        Returns:
            Array of fraud probabilities.

        Raises:
            RuntimeError: If probability prediction fails.
        """
        try:
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(X)[:, 1]

            if hasattr(self.model, "decision_function"):
                scores = self.model.decision_function(X)
                return 1 / (1 + np.exp(-scores))

            raise AttributeError("Model must support predict_proba or decision_function.")

        except Exception as exc:
            logger.exception("Failed to predict fraud probabilities.")
            raise RuntimeError("Fraud probability prediction failed.") from exc

    def get_risk_level(self, probability: float) -> str:
        """
        Convert fraud probability into a business-friendly risk level.

        Args:
            probability: Fraud probability between 0 and 1.

        Returns:
            Risk level: LOW, MEDIUM, HIGH, or CRITICAL.

        Raises:
            ValueError: If probability is outside the range [0, 1].
        """
        if probability < 0 or probability > 1:
            raise ValueError("Probability must be between 0 and 1.")

        if probability < 0.30:
            return "LOW"
        if probability < 0.60:
            return "MEDIUM"
        if probability < 0.85:
            return "HIGH"

        return "CRITICAL"

    def _get_top_factors(self, X_instance: pd.DataFrame, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Estimate top factors behind a single prediction.

        This dashboard-friendly method uses model feature importances combined
        with the absolute input feature values. For deeper explainability, the
        SHAP notebook and `src/explain.py` should be used.

        Args:
            X_instance: Single-row preprocessed transaction DataFrame.
            top_n: Number of factors to return.

        Returns:
            List of dictionaries containing top feature factors.
        """
        try:
            if len(X_instance) != 1:
                raise ValueError("X_instance must contain exactly one row.")

            if hasattr(self.model, "feature_importances_"):
                importances = np.asarray(self.model.feature_importances_)
            else:
                importances = np.ones(len(self.feature_names)) / len(self.feature_names)

            values = X_instance.iloc[0].to_numpy(dtype=float)
            impact_scores = np.abs(values) * importances

            factors_df = pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "value": values,
                    "importance": importances,
                    "impact_score": impact_scores,
                }
            )

            top_factors = (
                factors_df.sort_values("impact_score", ascending=False)
                .head(top_n)
                .to_dict(orient="records")
            )

            return top_factors

        except Exception as exc:
            logger.warning("Could not calculate top factors: %s", exc)
            return []

    def predict_single(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict fraud risk for a single transaction.

        Args:
            transaction: Dictionary containing transaction features.

        Returns:
            Dictionary with fraud probability, fraud flag, risk level, and
            top contributing factors.

        Raises:
            RuntimeError: If single prediction fails.
        """
        try:
            input_df = pd.DataFrame([transaction])
            processed_input = self.preprocess_input(input_df)

            probability = float(self._predict_probability(processed_input)[0])
            is_fraud = probability >= 0.50
            risk_level = self.get_risk_level(probability)
            top_factors = self._get_top_factors(processed_input)

            result = {
                "fraud_probability": probability,
                "is_fraud": is_fraud,
                "risk_level": risk_level,
                "top_factors": top_factors,
            }

            logger.info(
                "Single prediction completed. Probability=%.4f, Risk=%s.",
                probability,
                risk_level,
            )

            return result

        except Exception as exc:
            logger.exception("Failed to predict single transaction.")
            raise RuntimeError("Single transaction prediction failed.") from exc

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict fraud risk for a batch of transactions.

        Args:
            df: Raw or processed transaction DataFrame.

        Returns:
            Original DataFrame with added columns:
            `fraud_probability`, `is_fraud`, and `risk_level`.

        Raises:
            RuntimeError: If batch prediction fails.
        """
        try:
            if df.empty:
                raise ValueError("Batch input DataFrame cannot be empty.")

            output_df = df.copy()
            processed_df = self.preprocess_input(df)

            probabilities = self._predict_probability(processed_df)

            output_df["fraud_probability"] = probabilities
            output_df["is_fraud"] = output_df["fraud_probability"] >= 0.50
            output_df["risk_level"] = output_df["fraud_probability"].apply(self.get_risk_level)

            logger.info("Batch prediction completed for %s rows.", len(output_df))
            return output_df

        except Exception as exc:
            logger.exception("Failed to predict batch transactions.")
            raise RuntimeError("Batch transaction prediction failed.") from exc

    def get_threshold_metrics(self, df: pd.DataFrame, threshold: float) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1-score at a custom threshold.

        The input DataFrame must contain:

        - `Class`: actual labels where 0 = legitimate and 1 = fraud
        - `fraud_probability`: predicted fraud probabilities

        Args:
            df: DataFrame containing actual labels and fraud probabilities.
            threshold: Fraud classification threshold between 0 and 1.

        Returns:
            Dictionary with precision, recall, and F1-score.

        Raises:
            ValueError: If required columns are missing or threshold is invalid.
            RuntimeError: If metric calculation fails.
        """
        try:
            if threshold < 0 or threshold > 1:
                raise ValueError("Threshold must be between 0 and 1.")

            required_columns = {"Class", "fraud_probability"}
            missing_columns = required_columns.difference(df.columns)

            if missing_columns:
                raise ValueError(
                    f"Missing required columns for threshold metrics: {missing_columns}"
                )

            y_true = df["Class"]
            y_pred = (df["fraud_probability"] >= threshold).astype(int)

            metrics = {
                "threshold": float(threshold),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
            }

            logger.info("Threshold metrics calculated: %s.", metrics)
            return metrics

        except ValueError:
            logger.exception("Threshold metric validation failed.")
            raise
        except Exception as exc:
            logger.exception("Failed to calculate threshold metrics.")
            raise RuntimeError("Threshold metric calculation failed.") from exc

"""
Model training utilities for the PayGuard fraud detection project.

This module contains the ModelTrainer class, which trains baseline and advanced
machine learning models, evaluates classification performance, tunes XGBoost,
and saves or loads trained model artifacts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


class ModelTrainer:
    """
    Train, evaluate, compare, tune, save, and load fraud detection models.

    The class supports:

    - Logistic Regression baseline
    - Random Forest classifier
    - XGBoost classifier
    - Model evaluation using fraud-sensitive metrics
    - XGBoost hyperparameter tuning with GridSearchCV
    - Model persistence using joblib

    Args:
        random_state: Random seed for reproducible model training.
    """

    def __init__(self, random_state: int = 42) -> None:
        """
        Initialize the ModelTrainer.

        Args:
            random_state: Random seed used across supported models.
        """
        self.random_state = random_state
        logger.info("ModelTrainer initialized with random_state=%s.", random_state)

    def train_logistic_regression(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> LogisticRegression:
        """
        Train a Logistic Regression baseline model.

        Args:
            X: Training feature matrix.
            y: Training target vector.

        Returns:
            Trained LogisticRegression model.

        Raises:
            RuntimeError: If model training fails.
        """
        try:
            logger.info("Training Logistic Regression model.")

            model = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=-1,
            )

            start_time = perf_counter()
            model.fit(X, y)
            elapsed_time = perf_counter() - start_time

            logger.info(
                "Logistic Regression training completed in %.2f seconds.",
                elapsed_time,
            )

            return model

        except Exception as exc:
            logger.exception("Failed to train Logistic Regression model.")
            raise RuntimeError("Logistic Regression training failed.") from exc

    def train_random_forest(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> RandomForestClassifier:
        """
        Train a Random Forest classifier with 100 trees.

        Args:
            X: Training feature matrix.
            y: Training target vector.

        Returns:
            Trained RandomForestClassifier model.

        Raises:
            RuntimeError: If model training fails.
        """
        try:
            logger.info("Training Random Forest model.")

            model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
            )

            start_time = perf_counter()
            model.fit(X, y)
            elapsed_time = perf_counter() - start_time

            logger.info(
                "Random Forest training completed in %.2f seconds.",
                elapsed_time,
            )

            return model

        except Exception as exc:
            logger.exception("Failed to train Random Forest model.")
            raise RuntimeError("Random Forest training failed.") from exc

    def train_xgboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> XGBClassifier:
        """
        Train an XGBoost classifier.

        Args:
            X: Training feature matrix.
            y: Training target vector.

        Returns:
            Trained XGBClassifier model.

        Raises:
            RuntimeError: If model training fails.
        """
        try:
            logger.info("Training XGBoost model.")

            model = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1,
            )

            start_time = perf_counter()
            model.fit(X, y)
            elapsed_time = perf_counter() - start_time

            logger.info(
                "XGBoost training completed in %.2f seconds.",
                elapsed_time,
            )

            return model

        except Exception as exc:
            logger.exception("Failed to train XGBoost model.")
            raise RuntimeError("XGBoost training failed.") from exc

    def _get_positive_class_scores(
        self,
        model: Any,
        X_test: pd.DataFrame,
    ) -> np.ndarray:
        """
        Get positive-class probability scores from a trained classifier.

        Args:
            model: Trained classification model.
            X_test: Test feature matrix.

        Returns:
            Array of positive-class prediction scores.

        Raises:
            AttributeError: If the model does not support probability scoring.
        """
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X_test)[:, 1]

        if hasattr(model, "decision_function"):
            return model.decision_function(X_test)

        raise AttributeError("Model must support predict_proba or decision_function.")

    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        """
        Evaluate a trained model on test data.

        Args:
            model: Trained classifier.
            X_test: Test feature matrix.
            y_test: Test target vector.

        Returns:
            Dictionary containing evaluation metrics.

        Raises:
            RuntimeError: If model evaluation fails.
        """
        try:
            logger.info("Evaluating model: %s.", model.__class__.__name__)

            y_pred = model.predict(X_test)
            y_scores = self._get_positive_class_scores(model, X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_scores),
                "pr_auc": average_precision_score(y_test, y_scores),
            }

            logger.info("Model evaluation completed: %s.", metrics)

            return metrics

        except Exception as exc:
            logger.exception("Failed to evaluate model.")
            raise RuntimeError("Model evaluation failed.") from exc

    def compare_models(self, results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Create and print a side-by-side model comparison table.

        Args:
            results: Dictionary where keys are model names and values are
                metric dictionaries returned by evaluate_model.

        Returns:
            DataFrame containing model comparison metrics.

        Raises:
            RuntimeError: If comparison table creation fails.
        """
        try:
            logger.info("Creating model comparison table.")

            comparison_df = (
                pd.DataFrame.from_dict(results, orient="index")
                .reset_index()
                .rename(columns={"index": "model"})
            )

            numeric_columns = comparison_df.select_dtypes(include="number").columns
            comparison_df[numeric_columns] = comparison_df[numeric_columns].round(4)

            comparison_df = comparison_df.sort_values(
                by=["pr_auc", "f1_score", "recall"],
                ascending=False,
            )

            print("\nModel Comparison Table:")
            print(comparison_df.to_string(index=False))

            return comparison_df

        except Exception as exc:
            logger.exception("Failed to compare models.")
            raise RuntimeError("Model comparison failed.") from exc

    def tune_xgboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> GridSearchCV:
        """
        Tune XGBoost hyperparameters using GridSearchCV.

        The tuning process optimizes average precision, which is equivalent
        to PR-AUC and is useful for imbalanced fraud detection problems.

        Args:
            X: Training feature matrix.
            y: Training target vector.

        Returns:
            Fitted GridSearchCV object.

        Raises:
            RuntimeError: If hyperparameter tuning fails.
        """
        try:
            logger.info("Starting XGBoost hyperparameter tuning.")

            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            }

            model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1,
            )

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring="average_precision",
                cv=3,
                n_jobs=-1,
                verbose=1,
            )

            start_time = perf_counter()
            grid_search.fit(X, y)
            elapsed_time = perf_counter() - start_time

            logger.info(
                "XGBoost tuning completed in %.2f seconds. Best score: %.4f.",
                elapsed_time,
                grid_search.best_score_,
            )
            logger.info("Best parameters: %s.", grid_search.best_params_)

            return grid_search

        except Exception as exc:
            logger.exception("Failed to tune XGBoost model.")
            raise RuntimeError("XGBoost hyperparameter tuning failed.") from exc

    def save_model(self, model: Any, path: str | Path) -> None:
        """
        Save a trained model to disk using joblib.

        Args:
            model: Trained model object.
            path: Destination file path.

        Raises:
            RuntimeError: If saving the model fails.
        """
        try:
            model_path = Path(path)
            model_path.parent.mkdir(parents=True, exist_ok=True)

            joblib.dump(model, model_path)

            logger.info("Model saved successfully to %s.", model_path)

        except Exception as exc:
            logger.exception("Failed to save model.")
            raise RuntimeError("Model saving failed.") from exc

    def load_model(self, path: str | Path) -> Any:
        """
        Load a trained model from disk using joblib.

        Args:
            path: Path to the saved model file.

        Returns:
            Loaded model object.

        Raises:
            FileNotFoundError: If the model file does not exist.
            RuntimeError: If loading the model fails.
        """
        try:
            model_path = Path(path)

            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            model = joblib.load(model_path)

            logger.info("Model loaded successfully from %s.", model_path)

            return model

        except FileNotFoundError:
            logger.exception("Model file was not found.")
            raise
        except Exception as exc:
            logger.exception("Failed to load model.")
            raise RuntimeError("Model loading failed.") from exc


if __name__ == "__main__":
    processed_dir = Path("data/processed")
    model_path = Path("models/fraud_model.pkl")

    X_train = joblib.load(processed_dir / "X_train.pkl")
    X_test = joblib.load(processed_dir / "X_test.pkl")
    y_train = joblib.load(processed_dir / "y_train.pkl")
    y_test = joblib.load(processed_dir / "y_test.pkl")

    trainer = ModelTrainer(random_state=42)

    logistic = trainer.train_logistic_regression(X_train, y_train)
    random_forest = trainer.train_random_forest(X_train, y_train)
    xgboost = trainer.train_xgboost(X_train, y_train)

    results = {
        "Logistic Regression": trainer.evaluate_model(logistic, X_test, y_test),
        "Random Forest": trainer.evaluate_model(random_forest, X_test, y_test),
        "XGBoost": trainer.evaluate_model(xgboost, X_test, y_test),
    }

    trainer.compare_models(results)

    tuned_xgb = trainer.tune_xgboost(X_train, y_train)
    best_model = tuned_xgb.best_estimator_

    final_metrics = trainer.evaluate_model(best_model, X_test, y_test)
    print("\nFinal Tuned XGBoost Metrics:")
    print(final_metrics)

    trainer.save_model(best_model, model_path)

"""
SHAP explainability utilities for the PayGuard fraud detection project.

This module provides the FraudExplainer class for global and local model
explanations using SHAP. It is designed for XGBoost-based fraud detection
models but also handles common binary-class SHAP output formats returned by
tree-based scikit-learn models during testing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

plt.switch_backend("Agg")

PlotType = Literal["bar", "dot", "violin"]


class FraudExplainer:
    """
    Explain PayGuard fraud detection predictions using SHAP.

    The class supports:

    - Calculating SHAP values
    - Creating SHAP summary plots
    - Explaining individual predictions
    - Returning global feature importance as a DataFrame

    Args:
        model: Trained fraud detection model.
        X_train: Training feature matrix used as model background/reference data.

    Raises:
        RuntimeError: If the SHAP explainer cannot be initialized.
    """

    def __init__(self, model: Any, X_train: pd.DataFrame) -> None:
        """
        Initialize the FraudExplainer.

        Args:
            model: Trained model object, preferably an XGBoost classifier.
            X_train: Training feature DataFrame.

        Raises:
            ValueError: If X_train is empty.
            RuntimeError: If SHAP explainer initialization fails.
        """
        try:
            if X_train.empty:
                raise ValueError("X_train cannot be empty.")

            self.model = model
            self.X_train = X_train.copy()
            self.feature_names = list(self.X_train.columns)
            self.explainer = shap.TreeExplainer(self.model)

        except ValueError:
            raise
        except Exception as exc:
            raise RuntimeError("Failed to initialize FraudExplainer.") from exc

    @staticmethod
    def _normalise_shap_values(shap_values: Any) -> np.ndarray:
        """
        Convert SHAP output into a 2D array for the positive fraud class.

        SHAP can return different formats depending on model type and SHAP
        version:

        - list[class_0_values, class_1_values]
        - array with shape (n_samples, n_features)
        - array with shape (n_samples, n_features, n_classes)
        - array with shape (n_classes, n_samples, n_features)

        For binary fraud detection, this method always returns the positive
        fraud-class values as shape (n_samples, n_features).

        Args:
            shap_values: Raw SHAP values returned by the explainer.

        Returns:
            Normalised 2D SHAP values array.

        Raises:
            ValueError: If SHAP values cannot be normalised to 2D.
        """
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]

        values = np.asarray(shap_values)

        if values.ndim == 2:
            return values

        if values.ndim == 3:
            if values.shape[-1] <= 2:
                return values[:, :, -1]

            if values.shape[0] <= 2:
                return values[-1, :, :]

        raise ValueError(
            "Unsupported SHAP values shape. Expected 2D array or binary-class 3D array, "
            f"received shape {values.shape}."
        )

    def _get_expected_value(self) -> float:
        """
        Return the expected value for the positive fraud class.

        Returns:
            Expected SHAP base value as a float.
        """
        expected_value = self.explainer.expected_value

        if isinstance(expected_value, (list, np.ndarray)):
            expected_array = np.asarray(expected_value).ravel()
            return float(expected_array[-1])

        return float(expected_value)

    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate SHAP values for the provided feature matrix.

        Args:
            X: Feature matrix to explain.

        Returns:
            Array of SHAP values for the positive fraud class.

        Raises:
            ValueError: If X is empty.
            RuntimeError: If SHAP value calculation fails.
        """
        try:
            if X.empty:
                raise ValueError("X cannot be empty.")

            raw_shap_values = self.explainer.shap_values(X)
            return self._normalise_shap_values(raw_shap_values)

        except ValueError:
            raise
        except Exception as exc:
            raise RuntimeError("Failed to calculate SHAP values.") from exc

    def plot_summary(
        self,
        shap_values: np.ndarray,
        X: pd.DataFrame,
        plot_type: PlotType = "bar",
        output_path: str | Path = "reports/figures/shap_summary.png",
    ) -> Path:
        """
        Create and save a SHAP summary plot.

        Args:
            shap_values: SHAP values array.
            X: Feature matrix used for explanation.
            plot_type: SHAP summary plot type. Supported values are
                "bar", "dot", and "violin".
            output_path: Path where the plot should be saved.

        Returns:
            Path to the saved plot image.

        Raises:
            ValueError: If an unsupported plot type is provided.
            RuntimeError: If plot generation fails.
        """
        try:
            supported_plot_types = {"bar", "dot", "violin"}

            if plot_type not in supported_plot_types:
                raise ValueError(
                    f"Unsupported plot_type '{plot_type}'. Choose from {supported_plot_types}."
                )

            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)

            plt.figure(figsize=(10, 7))
            shap.summary_plot(
                shap_values,
                X,
                plot_type=plot_type,
                max_display=15,
                show=False,
            )
            plt.tight_layout()
            plt.savefig(output, dpi=300, bbox_inches="tight")
            plt.close()

            return output

        except ValueError:
            raise
        except Exception as exc:
            raise RuntimeError("Failed to create SHAP summary plot.") from exc

    def explain_prediction(self, X_instance: pd.DataFrame) -> Dict[str, Any]:
        """
        Explain one individual prediction and return the top 5 factors.

        Args:
            X_instance: Single-row feature DataFrame.

        Returns:
            Dictionary containing prediction, fraud probability, base value,
            model output value, and top 5 feature factors.

        Raises:
            ValueError: If X_instance does not contain exactly one row.
            RuntimeError: If local explanation fails.
        """
        try:
            if len(X_instance) != 1:
                raise ValueError("X_instance must contain exactly one row.")

            shap_values = self.get_shap_values(X_instance)[0]
            expected_value = self._get_expected_value()

            fraud_probability = float(self.model.predict_proba(X_instance)[:, 1][0])
            prediction = int(self.model.predict(X_instance)[0])

            factors_df = pd.DataFrame(
                {
                    "feature": X_instance.columns.tolist(),
                    "feature_value": X_instance.iloc[0].values,
                    "shap_value": shap_values,
                    "absolute_impact": np.abs(shap_values),
                }
            )

            top_factors = (
                factors_df.sort_values("absolute_impact", ascending=False)
                .head(5)
                .to_dict(orient="records")
            )

            model_output_value = float(expected_value + np.sum(shap_values))

            return {
                "prediction": prediction,
                "fraud_probability": fraud_probability,
                "base_value": expected_value,
                "model_output_value": model_output_value,
                "top_5_factors": top_factors,
            }

        except ValueError:
            raise
        except Exception as exc:
            raise RuntimeError("Failed to explain prediction.") from exc

    def get_feature_importance_df(self) -> pd.DataFrame:
        """
        Return global SHAP feature importance as a sorted DataFrame.

        SHAP importance is calculated as the mean absolute SHAP value for
        each feature across the training data.

        Returns:
            DataFrame sorted by SHAP importance in descending order.

        Raises:
            RuntimeError: If feature importance calculation fails.
        """
        try:
            shap_values = self.get_shap_values(self.X_train)
            shap_importance = np.abs(shap_values).mean(axis=0)

            importance_df = pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "shap_importance": shap_importance,
                }
            )

            importance_df = importance_df.sort_values(
                "shap_importance",
                ascending=False,
            ).reset_index(drop=True)

            return importance_df

        except Exception as exc:
            raise RuntimeError("Failed to calculate feature importance.") from exc

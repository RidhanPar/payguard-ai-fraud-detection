"""
Utility helpers for the PayGuard fraud detection project.

This module contains reusable helper functions for formatting currency,
loading sample processed data, and estimating simple business impact for the
Streamlit dashboard.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import joblib
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


Number = Union[int, float]


def format_currency(amount: Number, currency: str = "£") -> str:
    """
    Format a numeric amount as a currency string.

    Args:
        amount: Numeric amount to format.
        currency: Currency symbol to display. Defaults to British pound.

    Returns:
        Formatted currency string.

    Raises:
        ValueError: If amount cannot be converted to a number.
    """
    try:
        numeric_amount = float(amount)
        return f"{currency}{numeric_amount:,.2f}"

    except (TypeError, ValueError) as exc:
        logger.exception("Failed to format amount as currency.")
        raise ValueError("Amount must be a valid number.") from exc


def load_sample_data(n: int) -> pd.DataFrame:
    """
    Load random sample rows from the processed test set.

    This helper is useful for Streamlit demos where users want to test the
    trained model with existing processed transactions.

    Args:
        n: Number of random rows to return.

    Returns:
        DataFrame containing n sampled rows from X_test and the matching Class
        label when y_test is available.

    Raises:
        ValueError: If n is not positive.
        FileNotFoundError: If required processed files do not exist.
        RuntimeError: If sample loading fails.
    """
    try:
        if n <= 0:
            raise ValueError("n must be greater than 0.")

        project_root = Path.cwd()
        possible_roots = [project_root, project_root.parent]

        processed_dir = None
        for root in possible_roots:
            candidate = root / "data" / "processed"
            if (candidate / "X_test.pkl").exists():
                processed_dir = candidate
                break

        if processed_dir is None:
            raise FileNotFoundError(
                "Could not find data/processed/X_test.pkl. "
                "Please run notebooks/02_preprocessing.ipynb first."
            )

        X_test_path = processed_dir / "X_test.pkl"
        y_test_path = processed_dir / "y_test.pkl"

        X_test = joblib.load(X_test_path)

        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)

        sample_size = min(n, len(X_test))
        sample_df = X_test.sample(n=sample_size, random_state=42).copy()

        if y_test_path.exists():
            y_test = joblib.load(y_test_path)

            if not isinstance(y_test, pd.Series):
                y_test = pd.Series(y_test, index=X_test.index, name="Class")

            sample_df["Class"] = y_test.loc[sample_df.index].values

        logger.info("Loaded %s sample rows from processed test set.", len(sample_df))
        return sample_df.reset_index(drop=True)

    except (ValueError, FileNotFoundError):
        logger.exception("Sample data loading validation failed.")
        raise
    except Exception as exc:
        logger.exception("Failed to load sample data.")
        raise RuntimeError("Failed to load sample data.") from exc


def calculate_business_impact(fraud_caught: int, total_fraud_amount: Number) -> float:
    """
    Estimate money saved by catching fraudulent transactions.

    A simple business assumption is used: every caught fraud transaction
    prevents the corresponding fraud amount from becoming a financial loss.

    Args:
        fraud_caught: Number of fraudulent transactions caught by the model.
        total_fraud_amount: Total value of fraud caught.

    Returns:
        Estimated amount saved.

    Raises:
        ValueError: If inputs are invalid.
    """
    try:
        if fraud_caught < 0:
            raise ValueError("fraud_caught cannot be negative.")

        amount = float(total_fraud_amount)

        if amount < 0:
            raise ValueError("total_fraud_amount cannot be negative.")

        estimated_saved = amount

        logger.info(
            "Estimated business impact calculated. Fraud caught=%s, Saved=%.2f.",
            fraud_caught,
            estimated_saved,
        )

        return estimated_saved

    except (TypeError, ValueError) as exc:
        logger.exception("Failed to calculate business impact.")
        raise ValueError(
            "fraud_caught must be a non-negative integer and "
            "total_fraud_amount must be a non-negative number."
        ) from exc

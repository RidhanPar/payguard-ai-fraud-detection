"""
Project configuration constants for PayGuard.

This module stores all reusable file paths, model settings, and experiment
configuration values used across the PayGuard fraud detection pipeline.
Keeping these values centralized improves maintainability and prevents
hard-coded values from being repeated across multiple scripts.
"""

from typing import Final

DATA_PATH: Final[str] = "data/raw/creditcard.csv"
PROCESSED_PATH: Final[str] = "data/processed/cleaned_data.csv"
MODEL_PATH: Final[str] = "models/fraud_model.pkl"

RANDOM_STATE: Final[int] = 42
TEST_SIZE: Final[float] = 0.2
FRAUD_THRESHOLD: Final[float] = 0.5

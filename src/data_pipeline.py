"""
Data preprocessing pipeline for PayGuard fraud detection.

This module provides a production-ready data pipeline for loading, cleaning,
feature engineering, splitting, handling class imbalance, and saving processed
datasets for the PayGuard AI-powered payment fraud detection project.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


class FraudDataPipeline:
    """
    End-to-end preprocessing pipeline for payment fraud detection.

    The pipeline performs the following steps:

    1. Load raw transaction data from CSV.
    2. Clean missing values and duplicate rows.
    3. Scale `Amount` and `Time` using StandardScaler.
    4. Split the dataset into train and test sets using stratified sampling.
    5. Apply SMOTE to the training set only.
    6. Save processed splits and fitted scaler to disk.

    Args:
        config: Dictionary containing file paths and preprocessing parameters.

    Expected config keys:
        DATA_PATH: Path to the raw CSV file.
        PROCESSED_DIR: Directory where processed files should be saved.
        MODEL_DIR: Directory where model artifacts should be saved.
        RANDOM_STATE: Random seed for reproducibility.
        TEST_SIZE: Test split ratio.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the FraudDataPipeline.

        Args:
            config: Configuration dictionary with project paths and parameters.

        Raises:
            ValueError: If required configuration keys are missing.
        """
        self.config = config
        self.required_config_keys = {
            "DATA_PATH",
            "PROCESSED_DIR",
            "MODEL_DIR",
            "RANDOM_STATE",
            "TEST_SIZE",
        }

        missing_keys = self.required_config_keys.difference(self.config.keys())
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")

        self.data_path = Path(str(self.config["DATA_PATH"]))
        self.processed_dir = Path(str(self.config["PROCESSED_DIR"]))
        self.model_dir = Path(str(self.config["MODEL_DIR"]))
        self.random_state = int(self.config["RANDOM_STATE"])
        self.test_size = float(self.config["TEST_SIZE"])

        self.scaler = StandardScaler()

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info("FraudDataPipeline initialized successfully.")

    def load_data(self) -> pd.DataFrame:
        """
        Load raw transaction data from the configured CSV path.

        Returns:
            Raw transaction DataFrame.

        Raises:
            FileNotFoundError: If the configured CSV file does not exist.
            RuntimeError: If loading the CSV fails.
        """
        try:
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")

            df = pd.read_csv(self.data_path)
            logger.info(
                "Loaded data successfully from %s with shape %s.",
                self.data_path,
                df.shape,
            )
            return df

        except FileNotFoundError:
            logger.exception("Raw data file was not found.")
            raise
        except Exception as exc:
            logger.exception("Failed to load raw data.")
            raise RuntimeError("Failed to load raw transaction data.") from exc

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw transaction dataset.

        This method validates required columns, removes duplicate rows,
        drops rows with missing values, and resets the index.

        Args:
            df: Raw transaction DataFrame.

        Returns:
            Cleaned transaction DataFrame.

        Raises:
            ValueError: If required columns are missing.
            RuntimeError: If cleaning fails.
        """
        try:
            required_columns = {"Time", "Amount", "Class"}
            missing_columns = required_columns.difference(df.columns)

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            initial_rows = len(df)

            cleaned_df = df.drop_duplicates().dropna().reset_index(drop=True)

            removed_rows = initial_rows - len(cleaned_df)
            logger.info("Cleaned data successfully. Removed %s rows.", removed_rows)

            return cleaned_df

        except ValueError:
            logger.exception("Data validation failed during cleaning.")
            raise
        except Exception as exc:
            logger.exception("Failed to clean data.")
            raise RuntimeError("Failed to clean transaction data.") from exc

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer model-ready features from the cleaned dataset.

        This method standardizes the `Amount` and `Time` columns using
        StandardScaler, creates `Amount_Scaled` and `Time_Scaled`, and drops
        the original `Amount` and `Time` columns.

        Args:
            df: Cleaned transaction DataFrame.

        Returns:
            Feature-engineered DataFrame.

        Raises:
            ValueError: If required columns for feature engineering are missing.
            RuntimeError: If feature engineering fails.
        """
        try:
            required_columns = {"Amount", "Time", "Class"}
            missing_columns = required_columns.difference(df.columns)

            if missing_columns:
                raise ValueError(f"Missing required feature engineering columns: {missing_columns}")

            engineered_df = df.copy()

            engineered_df[["Amount_Scaled", "Time_Scaled"]] = self.scaler.fit_transform(
                engineered_df[["Amount", "Time"]]
            )

            engineered_df = engineered_df.drop(columns=["Amount", "Time"])

            scaler_path = self.model_dir / "scaler.pkl"
            joblib.dump(self.scaler, scaler_path)

            logger.info("Feature engineering completed successfully.")
            logger.info("Scaler saved to %s.", scaler_path)

            return engineered_df

        except ValueError:
            logger.exception("Feature engineering validation failed.")
            raise
        except Exception as exc:
            logger.exception("Failed to engineer features.")
            raise RuntimeError("Failed to engineer transaction features.") from exc

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split feature-engineered data into train and test sets.

        Stratified splitting preserves the fraud/legitimate class ratio
        across both training and testing datasets.

        Args:
            df: Feature-engineered DataFrame containing the `Class` target.

        Returns:
            Tuple containing X_train, X_test, y_train, and y_test.

        Raises:
            ValueError: If the target column is missing.
            RuntimeError: If splitting fails.
        """
        try:
            if "Class" not in df.columns:
                raise ValueError("Target column `Class` is missing.")

            X = df.drop(columns=["Class"])
            y = df["Class"]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y,
            )

            logger.info(
                "Train/test split completed. X_train: %s, X_test: %s.",
                X_train.shape,
                X_test.shape,
            )

            return X_train, X_test, y_train, y_test

        except ValueError:
            logger.exception("Data splitting validation failed.")
            raise
        except Exception as exc:
            logger.exception("Failed to split data.")
            raise RuntimeError("Failed to split transaction data.") from exc

    def apply_smote(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE to the training dataset.

        SMOTE is applied only to the training set to avoid test-data leakage.

        Args:
            X_train: Training feature matrix.
            y_train: Training target vector.

        Returns:
            Tuple containing resampled X_train and y_train.

        Raises:
            RuntimeError: If SMOTE resampling fails.
        """
        try:
            smote = SMOTE(random_state=self.random_state)

            X_train_resampled, y_train_resampled = smote.fit_resample(
                X_train,
                y_train,
            )

            logger.info(
                "SMOTE completed. Before: %s, After: %s.",
                X_train.shape,
                X_train_resampled.shape,
            )

            return X_train_resampled, y_train_resampled

        except Exception as exc:
            logger.exception("Failed to apply SMOTE.")
            raise RuntimeError("Failed to apply SMOTE to training data.") from exc

    def save_processed_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        """
        Save processed train and test datasets to disk.

        Args:
            X_train: Processed training features.
            X_test: Processed testing features.
            y_train: Processed training target.
            y_test: Testing target.

        Raises:
            RuntimeError: If saving processed files fails.
        """
        try:
            output_paths = {
                "X_train.pkl": X_train,
                "X_test.pkl": X_test,
                "y_train.pkl": y_train,
                "y_test.pkl": y_test,
            }

            for filename, data_object in output_paths.items():
                file_path = self.processed_dir / filename
                joblib.dump(data_object, file_path)
                logger.info("Saved %s successfully.", file_path)

        except Exception as exc:
            logger.exception("Failed to save processed data.")
            raise RuntimeError("Failed to save processed data files.") from exc

    def run_pipeline(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Run the full preprocessing pipeline end to end.

        Returns:
            Tuple containing final X_train, X_test, y_train, and y_test.
            The returned X_train and y_train are SMOTE-resampled.

        Raises:
            RuntimeError: If the full pipeline fails.
        """
        try:
            logger.info("Starting PayGuard preprocessing pipeline.")

            raw_df = self.load_data()
            cleaned_df = self.clean_data(raw_df)
            engineered_df = self.engineer_features(cleaned_df)

            X_train, X_test, y_train, y_test = self.split_data(engineered_df)
            X_train_resampled, y_train_resampled = self.apply_smote(
                X_train,
                y_train,
            )

            self.save_processed_data(
                X_train_resampled,
                X_test,
                y_train_resampled,
                y_test,
            )

            logger.info("PayGuard preprocessing pipeline completed successfully.")

            return X_train_resampled, X_test, y_train_resampled, y_test

        except Exception as exc:
            logger.exception("Pipeline execution failed.")
            raise RuntimeError("PayGuard preprocessing pipeline failed.") from exc


if __name__ == "__main__":
    pipeline_config = {
        "DATA_PATH": "data/raw/creditcard.csv",
        "PROCESSED_DIR": "data/processed",
        "MODEL_DIR": "models",
        "RANDOM_STATE": 42,
        "TEST_SIZE": 0.20,
    }

    pipeline = FraudDataPipeline(config=pipeline_config)
    pipeline.run_pipeline()

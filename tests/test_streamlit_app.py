"""Runtime smoke tests for the deployed Streamlit entry point."""

from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from app.dashboard import prepare_train_test_features


def test_streamlit_entrypoint_renders_without_exception() -> None:
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(str(app_path), default_timeout=60).run()

    assert not app.exception


def test_live_training_preprocessing_fits_scaler_on_training_rows_only() -> None:
    rng = np.random.default_rng(7)
    rows = 100
    df = pd.DataFrame(rng.normal(size=(rows, 28)), columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.arange(rows) * 100)
    df["Amount"] = np.linspace(1, 10_000, rows)
    df["Class"] = [0] * 90 + [1] * 10

    X_train, _, _, _, scaler, _ = prepare_train_test_features(df)

    assert np.isclose(X_train["Amount_Scaled"].mean(), 0.0)
    assert np.isclose(X_train["Time_Scaled"].mean(), 0.0)
    assert len(scaler.mean_) == 2

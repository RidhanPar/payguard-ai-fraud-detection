# Contributing

## Development Workflow

1. Create a focused branch from `main`.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run `pytest -q`.
4. Run `python -m black --check .`, `python -m isort --check .`, and `python -m flake8 . --max-line-length=100`.
5. Launch `streamlit run streamlit_app.py` and review the affected workflow.
6. Open a pull request describing the user impact, validation, and limitations.

Keep raw datasets, generated model artifacts, secrets, and customer data out of commits.

## Modeling Changes

Fit preprocessing only on training data, keep resampling out of evaluation data, report fraud-sensitive metrics, and document the operational impact of threshold changes.

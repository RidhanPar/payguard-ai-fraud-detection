.PHONY: test lint run clean

test:
	python -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=70

lint:
	python -m black --check .
	python -m isort --check .
	python -m flake8 . --max-line-length=100

run:
	python -m streamlit run app/dashboard.py

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', 'htmlcov', '.mypy_cache', '.ruff_cache', 'build', 'dist']]; [path.unlink() for path in pathlib.Path('.').rglob('*.pyc')]; [shutil.rmtree(path, ignore_errors=True) for path in pathlib.Path('.').rglob('__pycache__')]"

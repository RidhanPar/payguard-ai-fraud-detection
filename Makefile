.PHONY: test lint run clean deploy docker-build docker-run docker-compose

test:
	python -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=70

lint:
	python -m black --check .
	python -m isort --check .
	python -m flake8 . --max-line-length=100

run:
	python -m streamlit run streamlit_app.py

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', 'htmlcov', '.mypy_cache', '.ruff_cache', 'build', 'dist']]; [path.unlink() for path in pathlib.Path('.').rglob('*.pyc')]; [shutil.rmtree(path, ignore_errors=True) for path in pathlib.Path('.').rglob('__pycache__')]"

deploy:
	@echo "PayGuard Streamlit Cloud deployment"
	@echo ""
	@echo "1. Push your code to GitHub:"
	@echo "   git add ."
	@echo "   git commit -m \"Add deployment files\""
	@echo "   git push origin main"
	@echo ""
	@echo "2. Go to https://share.streamlit.io"
	@echo "3. Sign in with GitHub"
	@echo "4. Click New app"
	@echo "5. Select repository: RidhanPar/payguard-ai-fraud-detection"
	@echo "6. Select branch: main"
	@echo "7. Set main file path: streamlit_app.py"
	@echo "8. Click Deploy"
	@echo ""
	@echo "Full guide: docs/DEPLOYMENT.md"

docker-build:
	docker build -t payguard .

docker-run:
	docker run -p 8501:8501 payguard

docker-compose:
	docker compose up --build

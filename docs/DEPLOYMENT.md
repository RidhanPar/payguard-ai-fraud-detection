# PayGuard Deployment Guide

This guide explains how to deploy **PayGuard — AI-Powered Payment Fraud Detection** to Streamlit Community Cloud.

---

## 1. Push All Code to GitHub

Before deploying, make sure your project is pushed to GitHub.

### Include these files and folders

```text
app/
src/
tests/
notebooks/
docs/
.streamlit/
.github/
streamlit_app.py
requirements.txt
packages.txt
README.md
pyproject.toml
.flake8
Makefile
setup.py
```

### Exclude these files and folders in `.gitignore`

Do **not** push large datasets, generated model files, virtual environments, or local cache files.

```text
venv/
.venv/
env/

data/raw/*
data/processed/*
models/*
reports/
htmlcov/
.pytest_cache/
__pycache__/
*.pyc
.env
```

### Recommended Git commands

```bash
git status
git add .
git status
git commit -m "Add Streamlit deployment configuration"
git push origin main
```

Before committing, confirm that these are **not** staged:

```text
data/raw/creditcard.csv
data/processed/
models/
venv/
```

---

## 2. Go to Streamlit Community Cloud

Open:

```text
https://share.streamlit.io
```

or:

```text
https://streamlit.io/cloud
```

---

## 3. Sign in with GitHub

Click **Sign in with GitHub** and allow Streamlit to access your repository.

If your repository is private, make sure Streamlit has permission to access it.

---

## 4. Click "New app" or "Create app"

From your Streamlit Community Cloud workspace, click:

```text
New app
```

or:

```text
Create app
```

The wording may vary slightly depending on the current Streamlit interface.

---

## 5. Select Repository, Branch, and Main File Path

Use these settings:

| Field | Value |
|---|---|
| Repository | `RidhanPar/payguard-ai-fraud-detection` |
| Branch | `main` |
| Main file path | `streamlit_app.py` |

The root-level `streamlit_app.py` file imports and runs:

```text
app/dashboard.py
```

---

## 6. Click Deploy

Click:

```text
Deploy
```

Streamlit will install dependencies from:

```text
requirements.txt
```

If system-level packages are needed, Streamlit will also read:

```text
packages.txt
```

For this project, no extra apt/system packages are required.

---

## 7. Live App URL

After deployment, your app will be live at a URL similar to:

```text
https://your-username-payguard-streamlit-app-xyz.streamlit.app
```

You can add this link to:

```text
README.md
Portfolio website
LinkedIn project section
GitHub repository About section
```

---

# Important Deployment Note About Model Files

This project's `.gitignore` excludes:

```text
models/
data/raw/
data/processed/
```

That is correct for GitHub because datasets and generated models should not usually be pushed.

However, on Streamlit Cloud, the dashboard will not have your local trained files unless you provide them.

The app already handles this safely:

- If `models/fraud_model.pkl` is missing, it shows a **Model not trained yet** warning.
- If `models/scaler.pkl` is missing, it shows setup instructions.
- The app will not crash just because model files are missing.

For a live demo, you have three options:

## Option A — Demo mode

Keep model files excluded. The app will show a setup guide.

Good for showing project structure, code quality, and deployment readiness.

## Option B — Add small demo model files

Train a smaller demo model and include it only if file size is acceptable.

Before doing this, check GitHub file size limits and avoid committing very large files.

## Option C — Use external model storage

Store model files in external storage such as:

- Hugging Face Hub
- AWS S3
- Google Cloud Storage
- GitHub Releases

Then update the app to download the model at startup.

This is the most production-style approach.

---

# Common Deployment Errors and Fixes

## Error 1: `ModuleNotFoundError`

### Cause

A required Python package is missing from `requirements.txt`.

### Fix

Install it locally and add it to `requirements.txt`.

```bash
pip install package-name
pip freeze > requirements.txt
```

Then commit and push again.

---

## Error 2: App cannot find `models/fraud_model.pkl`

### Cause

The model file is not pushed to GitHub because `models/` is ignored.

### Fix

This is expected. The dashboard should show the setup guide instead of crashing.

For a full demo, use one of these options:

```text
1. Include a small demo model
2. Download model from external storage
3. Keep setup-guide mode
```

---

## Error 3: App cannot find `data/processed/X_test.pkl`

### Cause

Processed data files are generated locally and ignored by Git.

### Fix

Run preprocessing locally:

```bash
python src/data_pipeline.py
```

For Streamlit Cloud, either:

```text
1. Keep app in setup-guide mode
2. Add sample demo data
3. Load demo data from external storage
```

---

## Error 4: `streamlit_app.py` not found

### Cause

The main file path is wrong in Streamlit Cloud settings.

### Fix

Use:

```text
streamlit_app.py
```

not:

```text
app/dashboard.py
```

---

## Error 5: Dependency build fails

### Cause

Some package versions may not support the Python version selected by Streamlit Cloud.

### Fix

Use a stable Python version in Streamlit advanced settings, such as:

```text
Python 3.10
```

Also make sure your dependencies are compatible.

---

## Error 6: App starts locally but not on Streamlit Cloud

### Cause

Local-only files are being used, such as:

```text
C:\Users\This PC\Downloads\...
```

### Fix

Use relative project paths only.

Correct:

```python
Path(__file__).resolve().parents[1]
```

Avoid hard-coded local paths.

---

# Local Deployment Test

Before deploying, test locally:

```bash
streamlit run streamlit_app.py
```

or:

```bash
python -m streamlit run streamlit_app.py
```

Then open the local URL shown in the terminal.

---

# Final Checklist

Before deployment:

- [ ] GitHub repo is public or Streamlit has access.
- [ ] `requirements.txt` is present.
- [ ] `streamlit_app.py` is in the root folder.
- [ ] `.streamlit/config.toml` is present.
- [ ] `.gitignore` excludes data, models, venv, and cache files.
- [ ] App does not depend on local absolute paths.
- [ ] App handles missing model files gracefully.
- [ ] GitHub Actions CI passes.

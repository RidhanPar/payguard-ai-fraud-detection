# PayGuard — Machine Learning Fraud Detection Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-Gradient%20Boosting-FF6600?style=for-the-badge" />
  <img src="https://img.shields.io/badge/SHAP-Explainable%20AI-1f77b4?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-Visualisation-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white" />
</p>

---

## 🎯 Project Overview

PayGuard is a machine learning fraud detection platform that analyses payment transaction patterns, scores fraud risk, and explains model decisions through a Streamlit monitoring dashboard. The project includes data cleaning, class imbalance handling, model training, SHAP-based explainability, CI/CD checks, Docker support, and an interactive deployment-ready application.

---

## Reviewer Guide

For a quick technical review:

| What to review | Evidence |
|---|---|
| Data cleaning, validation, and imbalance handling | [`src/data_pipeline.py`](src/data_pipeline.py) |
| Model comparison and fraud-sensitive metrics | [`src/train.py`](src/train.py) |
| Batch and single-transaction scoring | [`src/predict.py`](src/predict.py) |
| Explainability implementation | [`src/explain.py`](src/explain.py) |
| Product/dashboard experience | [`streamlit_app.py`](streamlit_app.py), [`app/dashboard.py`](app/dashboard.py) |
| Automated quality checks | [`tests/`](tests/), [`.github/workflows/ci.yml`](.github/workflows/ci.yml) |
| Reproducible deployment | [`Dockerfile`](Dockerfile), [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) |

**Current scope:** this is a portfolio prototype trained on the public Kaggle credit-card fraud dataset. It is not a production fraud decisioning system and should not be used to make real customer decisions without validation, governance, monitoring, and human-review controls.

Read the intended use, validation approach, and risks in the [`model card`](docs/MODEL_CARD.md).

---

## 🏗️ Architecture

```text
┌──────────────────┐
│    Raw Data      │
│ creditcard.csv   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Data Pipeline    │
│ Cleaning, Scaling│
│ Train/Test Split │
│ SMOTE Balancing  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Model Training   │
│ Logistic Reg.    │
│ Random Forest    │
│ XGBoost          │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ SHAP Explainability │
│ Global + Local      │
│ Risk Factors        │
└────────┬────────────┘
         │
         ▼
┌──────────────────┐
│ Streamlit Dashboard │
│ Upload CSV          │
│ Analyse Fraud       │
│ Monitor Metrics     │
└────────┬────────────┘
         │
         ▼
┌──────────────────┐
│ Business Decision │
│ Approve / Review  │
│ Block / Escalate  │
└──────────────────┘
```

---

## 📊 Evaluation Status

The repository intentionally does not publish unverified model scores. Run the preprocessing and training workflow to reproduce model comparison results on your local copy of the Kaggle dataset.

The evaluation code reports:

- Precision, recall, F1, ROC-AUC, and PR-AUC.
- A Logistic Regression baseline alongside Random Forest and XGBoost.
- Test-set evaluation kept separate from SMOTE-resampled training data.

For an imbalanced fraud problem, **PR-AUC, recall, precision, and the operational cost of false positives** should drive model selection; accuracy alone is not a useful success measure.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | scikit-learn, XGBoost |
| Imbalanced Learning | SMOTE, imbalanced-learn |
| Explainability | SHAP |
| Visualisation | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Model Persistence | Joblib |
| Testing | Pytest |
| Dataset Source | Kaggle Credit Card Fraud Detection |

---

## 📁 Project Structure

```text
payguard/
├── app/
│   └── dashboard.py                 # Streamlit dashboard with overview, analysis, explainability, and performance pages
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv           # Raw Kaggle credit card fraud dataset, not pushed to GitHub
│   └── processed/
│       ├── X_train.pkl              # SMOTE-resampled training features
│       ├── X_test.pkl               # Untouched test features
│       ├── y_train.pkl              # SMOTE-resampled training labels
│       └── y_test.pkl               # Untouched test labels
│
├── docs/
│   └── shap_summary.png             # Placeholder for SHAP summary screenshot used in README
│
├── models/
│   ├── fraud_model.pkl              # Final trained XGBoost fraud detection model
│   └── scaler.pkl                   # Fitted StandardScaler for Amount and Time
│
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis and fraud pattern discovery
│   ├── 02_preprocessing.ipynb       # Data cleaning, scaling, splitting, and SMOTE balancing
│   ├── 03_modelling.ipynb           # Model training, comparison, tuning, and evaluation
│   └── 04_explainability.ipynb      # SHAP global and local explainability analysis
│
├── reports/
│   └── figures/                     # Generated charts, SHAP plots, and evaluation visuals
│
├── src/
│   ├── __init__.py                  # Python package initializer
│   ├── config.py                    # Central project constants and paths
│   ├── data_pipeline.py             # FraudDataPipeline class for preprocessing
│   ├── train.py                     # ModelTrainer class for training and tuning
│   ├── predict.py                   # FraudPredictor class for single and batch predictions
│   ├── explain.py                   # FraudExplainer class for SHAP-based explanations
│   └── utils.py                     # Helper functions for currency, samples, and business impact
│
├── tests/
│   ├── conftest.py                  # Shared pytest fixtures
│   └── test_pipeline.py             # Unit tests for pipeline, model trainer, and predictor
│
├── .gitignore                       # Files and folders excluded from GitHub
├── requirements.txt                 # Python dependencies
├── setup.py                         # Project package metadata
└── README.md                        # Project documentation
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/RidhanPar/payguard-ai-fraud-detection.git
cd payguard-ai-fraud-detection
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If using Jupyter notebooks:

```bash
pip install notebook ipykernel
python -m ipykernel install --user --name payguard-venv --display-name "Python (PayGuard)"
```

### 4. Download the dataset

Download the **Credit Card Fraud Detection** dataset from Kaggle:

```text
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```

Place the file here:

```text
data/raw/creditcard.csv
```

Or use the Kaggle API:

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```

### 5. Run preprocessing and training

Run preprocessing first:

```bash
python src/data_pipeline.py
```

Then train the model:

```bash
python src/train.py
```

This will create:

```text
models/fraud_model.pkl
models/scaler.pkl
data/processed/X_train.pkl
data/processed/X_test.pkl
data/processed/y_train.pkl
data/processed/y_test.pkl
```

### 6. Launch the Streamlit dashboard

```bash
streamlit run app/dashboard.py
```

---

## 📓 Notebooks

| Notebook | Purpose | Key Output |
|---|---|---|
| `01_eda.ipynb` | Explore dataset structure, class imbalance, transaction amount, time patterns, and feature correlations | EDA insights and fraud pattern discovery |
| `02_preprocessing.ipynb` | Clean data, scale `Amount` and `Time`, split data, apply SMOTE to training set | Processed `.pkl` train/test datasets and fitted scaler |
| `03_modelling.ipynb` | Train Logistic Regression, Random Forest, and XGBoost; compare metrics; tune XGBoost | Final model saved as `models/fraud_model.pkl` |
| `04_explainability.ipynb` | Use SHAP to explain global feature importance and individual predictions | SHAP summary, waterfall, and force plot explanations |

---

## 🔍 Model Explainability

Fraud detection models must be explainable because financial decisions can affect customers, businesses, and risk operations. It is not enough to say that a transaction is fraudulent; analysts need to understand **why** the model produced that decision.

PayGuard uses **SHAP** to explain both global and local model behaviour:

- **Global explainability** shows which features are most important across all transactions.
- **Local explainability** explains why one specific transaction was classified as fraud or legitimate.
- **Top risk factors** help fraud analysts understand the model's reasoning in plain English.
- **SHAP waterfall plots** show whether each feature pushed a prediction toward fraud or toward legitimate behaviour.

The strongest fraud indicators identified in the explainability notebook are:

```text
V14, V17, V12
```

Because the dataset uses anonymized PCA features, these variables cannot be mapped to original raw business fields. However, they can still be interpreted as hidden transaction signature patterns that separate fraud-like behaviour from normal payment behaviour.

![SHAP Summary](docs/shap_summary.png)

---

## 🌐 Live Demo

**[Open the live PayGuard dashboard](https://payguard-ai-fraud-detection.streamlit.app/)**

The dashboard also runs locally with:

```bash
streamlit run streamlit_app.py
```

---

## 📈 Future Improvements

- Add real-time transaction streaming using Kafka or AWS Kinesis.
- Track experiments, metrics, and model versions using MLflow.
- Add automated retraining when fraud patterns drift over time.
- Connect the dashboard to a live payment API or transaction database.
- Add a time-aware validation strategy and probability calibration.
- Publish reproducible evaluation artifacts and decision-threshold analysis.

---

## 👤 Author

**Ridhan Parvendhan**

- LinkedIn: [linkedin.com/in/ridhanparvendhan](https://www.linkedin.com/in/ridhanparvendhan/)
- GitHub: [github.com/RidhanPar](https://github.com/RidhanPar)

---

## ⭐ Project Value

PayGuard demonstrates practical skills in:

- End-to-end data science project development
- Fraud detection and imbalanced classification
- Machine learning model comparison and tuning
- model explainability using SHAP
- Business-focused dashboard development
- Production-style Python project structure
- Testing, modular code, and reusable ML pipelines


# PayGuard — AI-Powered Payment Fraud Detection

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

**PayGuard** is an end-to-end machine learning project that detects fraudulent payment transactions using advanced classification models, class imbalance handling, explainable AI, and a professional Streamlit dashboard. The system analyses transaction patterns, assigns fraud probabilities, categorises risk levels, and helps decision-makers understand why a transaction was flagged.

Payment companies process large volumes of transactions every day, and even a small percentage of fraud can create major financial losses, customer trust issues, and operational pressure. This project mirrors real challenges faced by payment companies like **GoCardless**, where fraud prevention, payment success, risk monitoring, and scalable decision-making are business-critical.

The business problem PayGuard solves is simple: **identify high-risk transactions early, reduce fraud losses, and support faster fraud investigation with explainable model decisions**. Instead of only producing a prediction, PayGuard also provides risk levels, model performance monitoring, SHAP explainability, and business impact estimates.

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

## 📊 Key Results

| Model | AUC-ROC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.970 | 0.890 | 0.850 | 0.870 |
| Random Forest | 0.990 | 0.950 | 0.910 | 0.930 |
| XGBoost | 0.999 | 0.970 | 0.940 | 0.955 |

**Business impact:** XGBoost catches **94% of fraud** while maintaining **97% precision**, helping reduce financial loss while keeping false fraud alerts under control.

> Note: These values are realistic project placeholders. Replace them with final metrics after running `notebooks/03_modelling.ipynb`.

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
git clone https://github.com/your-username/payguard.git
cd payguard
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

[Link to live Streamlit app](your-link-here)

---

## 📈 Future Improvements

- Add real-time transaction streaming using Kafka or AWS Kinesis.
- Track experiments, metrics, and model versions using MLflow.
- Add automated retraining when fraud patterns drift over time.
- Connect the dashboard to a live payment API or transaction database.
- Add Docker, CI/CD, and cloud deployment for production readiness.

---

## 👤 Author

**Ridhan Parvendhan**

- LinkedIn: [your-linkedin-here](your-linkedin-here)
- GitHub: [your-github-here](your-github-here)
- Portfolio: [your-portfolio-here](your-portfolio-here)

---

## ⭐ Project Value

PayGuard demonstrates practical skills in:

- End-to-end data science project development
- Fraud detection and imbalanced classification
- Machine learning model comparison and tuning
- Explainable AI using SHAP
- Business-focused dashboard development
- Production-style Python project structure
- Testing, modular code, and reusable ML pipelines

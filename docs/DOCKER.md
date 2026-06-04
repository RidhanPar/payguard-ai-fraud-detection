# PayGuard Docker Guide

This guide explains how to run **PayGuard — AI-Powered Payment Fraud Detection** using Docker.

Docker packages the Python environment, dependencies, and Streamlit server into one reproducible container.

---

## Prerequisites

Install Docker Desktop and confirm it works:

```bash
docker --version
docker compose version
```

---

## 1. Build the Docker Image

From the PayGuard project root:

```bash
docker build -t payguard .
```

---

## 2. Run the Streamlit App

```bash
docker run -p 8501:8501 payguard
```

Open:

```text
http://localhost:8501
```

---

## 3. Run the Full Stack with Docker Compose

```bash
docker compose up
```

This starts:

| Service | Purpose | Port |
|---|---|---:|
| `app` | PayGuard Streamlit dashboard | `8501` |
| `mlflow` | Optional MLflow tracking server | `5000` |

Open:

```text
http://localhost:8501
```

MLflow:

```text
http://localhost:5000
```

---

## 4. Run in Detached Mode

```bash
docker compose up -d
```

Stop services:

```bash
docker compose down
```

---

## 5. Rebuild After Changes

```bash
docker compose up --build
```

or:

```bash
docker build -t payguard .
```

---

## 6. Model and Data Files

The Docker Compose setup mounts these folders:

```text
./data   -> /app/data
./models -> /app/models
./reports -> /app/reports
```

If your local model files exist, Docker can access:

```text
models/fraud_model.pkl
models/scaler.pkl
data/processed/X_test.pkl
data/processed/y_test.pkl
```

If model files do not exist, the Streamlit dashboard shows a setup guide instead of crashing.

---

## 7. Common Docker Errors and Fixes

### Docker is not recognized

Open Docker Desktop and wait until Docker is running.

### Port 8501 is already in use

Run on another local port:

```bash
docker run -p 8502:8501 payguard
```

Then open:

```text
http://localhost:8502
```

### Model not trained yet

Run locally:

```bash
python src/data_pipeline.py
python src/train.py
```

Then restart Docker:

```bash
docker compose up --build
```

### XGBoost library issue

The Dockerfile installs `libgomp1`, which is commonly required by XGBoost on slim Linux images.

Rebuild without cache:

```bash
docker build --no-cache -t payguard .
```

---

## Useful Commands

```bash
docker build -t payguard .
docker run -p 8501:8501 payguard
docker compose up
docker compose up --build
docker compose up -d
docker compose down
docker ps
docker logs payguard-streamlit
```

---

## Makefile Shortcuts

```bash
make docker-build
make docker-run
make docker-compose
```

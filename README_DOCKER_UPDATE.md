# README Docker Update

Add this badge into the existing badge section near the top of `README.md`:

```html
<img src="https://img.shields.io/badge/Docker-Containerised-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
```

Add this section to `README.md` after **Quick Start** or before **Notebooks**:

---

## 🐳 Docker

PayGuard includes Docker support for consistent local and production-like execution.

### Build the Docker image

```bash
docker build -t payguard .
```

### Run the Streamlit app

```bash
docker run -p 8501:8501 payguard
```

Open:

```text
http://localhost:8501
```

### Run the full stack with Docker Compose

```bash
docker compose up --build
```

This starts:

| Service | Port | Purpose |
|---|---:|---|
| PayGuard Streamlit App | 8501 | Fraud detection dashboard |
| MLflow Tracking Server | 5000 | Optional experiment tracking |

### Makefile shortcuts

```bash
make docker-build
make docker-run
make docker-compose
```

Full Docker instructions are available in:

```text
docs/DOCKER.md
```

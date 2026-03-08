# 🌿 PlantMD Pro — Industry-Level Plant Disease Detection System

[![CI/CD](https://github.com/your-org/plantmd-pro/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/plantmd-pro/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade plant disease classification system powered by **EfficientNetV2** deep learning.
Built with FastAPI, Streamlit, Redis caching, JWT auth, Grad-CAM explainability, and Docker.

---

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Streamlit UI   │───▶│   FastAPI API   │───▶│  EfficientNetV2 │
│  (Port 8501)    │    │  (Port 8000)    │    │   ML Model      │
└─────────────────┘    └────────┬────────┘    └─────────────────┘
                                │
                       ┌────────▼────────┐
                       │  Redis Cache    │
                       │  (Port 6379)    │
                       └─────────────────┘
```

## ✨ Features

| Feature | Details |
|---|---|
| **Model** | EfficientNetV2-S with ImageNet transfer learning |
| **Accuracy** | ~95%+ on PlantVillage dataset |
| **Explainability** | Grad-CAM heatmap overlays |
| **API** | FastAPI with async endpoints, auto-docs, JWT auth |
| **Caching** | Redis with in-memory fallback |
| **Validation** | File type, size, magic bytes, dimensions |
| **Containerization** | Multi-stage Docker builds |
| **CI/CD** | GitHub Actions: lint → test → security scan → build |
| **Testing** | Pytest with 70%+ coverage enforced |
| **Monitoring** | Structured JSON logging with structlog |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (optional but recommended)
- Git

### Option 1: Docker (Recommended)

```bash
# Clone the repo
git clone https://github.com/your-org/plantmd-pro.git
cd plantmd-pro

# Copy environment file
cp .env.example .env

# Start all services
make docker-up

# Access:
# API Docs:  http://localhost:8000/docs
# Frontend:  http://localhost:8501
# Health:    http://localhost:8000/health
```

### Option 2: Local Development (PyCharm)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and configure environment
cp .env.example .env
# Edit .env with your settings

# 4. Start backend API
make dev
# OR: cd backend && uvicorn main:app --reload

# 5. Start frontend (new terminal)
make frontend
# OR: cd frontend && streamlit run app.py
```

---

## 🧠 Training Your Own Model

```bash
# 1. Download PlantVillage dataset from Kaggle:
# https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

# 2. Extract to data/raw/plantvillage/

# 3. Train (Phase 1: feature extraction, Phase 2: fine-tuning)
python ml_pipeline/train.py \
    --dataset data/raw/plantvillage \
    --output models/ \
    --epochs-phase1 10 \
    --epochs-phase2 20 \
    --batch-size 32

# Model saved to: models/plantmd_efficientnet.h5
# Labels saved to: models/labels.json
```

---

## 📡 API Reference

### POST `/api/v1/predict`
Upload a leaf image for disease prediction.

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -F "file=@leaf.jpg" \
  -F "generate_heatmap=false"
```

**Response:**
```json
{
  "display_name": "Tomato Late Blight",
  "confidence": 0.973,
  "is_healthy": false,
  "severity": "HIGH",
  "treatment": {
    "immediate_action": "Remove and destroy infected parts",
    "fungicide": "Chlorothalonil 2.5g/L",
    "prevention": "Improve air circulation"
  },
  "inference_ms": 47.3,
  "model_version": "2.0.0"
}
```

### GET `/api/v1/predict/classes`
List all 38 supported disease classes.

### POST `/api/v1/auth/token`
Get JWT token (username: `demo`, password: `demo` for testing).

### GET `/health`
Full health check with model status.

Full interactive docs: `http://localhost:8000/docs`

---

## 🧪 Running Tests

```bash
make test
# OR
cd backend && pytest tests/ -v --cov=. --cov-report=html

# View coverage report
open backend/htmlcov/index.html
```

---

## 📁 Project Structure

```
plantmd-pro/
├── backend/
│   ├── main.py                    # FastAPI application
│   ├── core/
│   │   ├── config.py              # Pydantic settings
│   │   ├── exceptions.py          # Custom exceptions
│   │   └── logging.py             # Structured logging
│   ├── api/routes/
│   │   ├── predict.py             # Prediction endpoint
│   │   ├── health.py              # Health checks
│   │   └── auth.py                # JWT authentication
│   ├── ml/
│   │   ├── model.py               # EfficientNetV2 + Grad-CAM
│   │   └── preprocessor.py        # Image validation & preprocessing
│   ├── services/
│   │   └── cache_service.py       # Redis caching
│   └── tests/
│       └── test_api.py            # Comprehensive test suite
├── frontend/
│   └── app.py                     # Streamlit UI
├── ml_pipeline/
│   └── train.py                   # Training pipeline
├── infrastructure/
│   ├── docker/
│   │   ├── Dockerfile.backend     # Multi-stage backend image
│   │   └── Dockerfile.frontend    # Frontend image
│   └── kubernetes/
│       └── deployment.yaml        # K8s manifests + HPA
├── .github/workflows/
│   └── ci.yml                     # GitHub Actions CI/CD
├── docker-compose.yml
├── requirements.txt
├── Makefile
├── .env.example
└── .pre-commit-config.yaml
```

---

## 🔒 Security

- JWT-based authentication
- File magic byte validation (prevents polyglot attacks)
- File size and dimension limits
- Non-root Docker user
- Secrets via environment variables (never hardcoded)
- CORS configured

---

## 📊 Supported Diseases (38 classes)

Tomato (9 diseases), Potato (3), Pepper (2), Corn/Maize (4), Apple (4), Grape (4), and more.
All healthy variants are also classified.

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Run `pre-commit install` for automatic code formatting
4. Make your changes and run `make test`
5. Submit a PR

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

**Built with ❤️ by the PlantMD Team | Dataset: PlantVillage (Penn State)**

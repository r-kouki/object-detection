# YOLO11 Object Detection - MLOps Pipeline

A **fully dockerized**, production-ready MLOps pipeline for object detection using Ultralytics YOLO11. Zero local ML tool installation required - everything runs in containers.

[![CI/CD](https://github.com/YOUR_USERNAME/object-detection/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/YOUR_USERNAME/object-detection/actions/workflows/ci-cd.yml)

## 📚 Documentation

- **[Architecture & Justifications](docs/ARCHITECTURE.md)** - Detailed architecture and technology choices
- **[README](README.md)** - Quick start guide

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Docker Compose Stack                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────────┐  │
│  │    MinIO    │    │  PostgreSQL │    │          MLflow Server          │  │
│  │  (S3 store) │◄───┤  (metadata) │◄───┤  (tracking + model registry)   │  │
│  │  :9000/9001 │    │    :5432    │    │           :5000                 │  │
│  └──────┬──────┘    └─────────────┘    └──────────────┬──────────────────┘  │
│         │                                              │                     │
│         │  ┌──────────────────────────────────────────┘                     │
│         │  │                                                                 │
│         ▼  ▼                                                                 │
│  ┌─────────────────────────────────┐    ┌────────────────────────────────┐  │
│  │       Trainer Container         │    │        API Container           │  │
│  │  DVC + ZenML + Optuna + YOLO    │    │   FastAPI /predict endpoint    │  │
│  │  (training & data versioning)   │    │      (model serving)           │  │
│  └─────────────────────────────────┘    └────────────────────────────────┘  │
│                                                      :8000                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start (100% Docker)

### Prerequisites
- Docker & Docker Compose (v2.0+)
- ~10GB disk space
- No local Python/ML tools needed!

### 1. Clone & Setup

```bash
git clone <your-repo-url>
cd object-detection

# Copy environment file
cp .env.example .env
```

### 2. Start Infrastructure

```bash
# Start all services (MinIO, PostgreSQL, MLflow)
docker compose up -d minio minio-init postgres mlflow

# Wait for services to be healthy (~30 seconds)
docker compose ps

# Verify buckets created
docker compose logs minio-init
```

**Access UIs:**
- MLflow: http://localhost:5000
- MinIO Console: http://localhost:9001 (admin: minioadmin/minioadmin)

### 3. Download & Version Dataset

```bash
# Download COCO128 and setup DVC
docker compose run --rm trainer python -m src.data.ingest --setup-dvc

# Push data to MinIO (version control)
docker compose run --rm trainer dvc push
```

### 4. Train Baseline Model (v1)

```bash
# Run full pipeline: ingest → train → eval → register
docker compose run --rm trainer python -m src.pipelines.zenml_pipeline \
    --config /app/configs/train_baseline.yaml

# Check MLflow UI for results: http://localhost:5000
```

### 5. Train Improved Model (v2)

```bash
docker compose run --rm trainer python -m src.pipelines.zenml_pipeline \
    --config /app/configs/train_v2.yaml
```

### 6. Run Optuna Hyperparameter Search

```bash
docker compose run --rm trainer python -m src.train.optuna_tune \
    --config /app/configs/optuna.yaml \
    --train-best
```

### 7. Deploy API

```bash
# Deploy with v1 model
MODEL_URI=models:/detector/1 docker compose up -d api

# Test health
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict \
    -F "file=@test_image.jpg"
```

### 8. Model Upgrade & Rollback Demo

```bash
# === DEPLOY v1 ===
MODEL_URI=models:/detector/1 docker compose up -d api
curl http://localhost:8000/model/info
# Save prediction output for comparison

# === UPGRADE TO v2 ===
docker compose down api
MODEL_URI=models:/detector/2 docker compose up -d api
curl http://localhost:8000/model/info
# Compare predictions - should see different results

# === ROLLBACK TO v1 ===
docker compose down api
MODEL_URI=models:/detector/1 docker compose up -d api
curl http://localhost:8000/model/info
# Verify same results as original v1
```

## 📁 Project Structure

```
/
├── docker-compose.yml          # Complete Docker stack
├── docker/
│   ├── trainer.Dockerfile      # Training container
│   └── api.Dockerfile          # Inference container
├── configs/
│   ├── train_baseline.yaml     # v1 baseline config
│   ├── train_v2.yaml           # v2 improved config
│   ├── optuna.yaml             # Hyperparameter search
│   └── serving.yaml            # API configuration
├── src/
│   ├── data/ingest.py          # Dataset download & DVC setup
│   ├── train/
│   │   ├── train.py            # YOLO training + MLflow
│   │   ├── eval.py             # Model evaluation
│   │   └── optuna_tune.py      # Hyperparameter optimization
│   ├── pipelines/zenml_pipeline.py  # Orchestration
│   ├── registry/register.py    # MLflow model registry
│   └── common/repro.py         # Reproducibility utils
├── api/
│   ├── main.py                 # FastAPI app
│   └── inference.py            # YOLO inference engine
├── tests/
│   ├── test_smoke_train.py     # Training smoke test
│   └── test_api.py             # API tests
├── .gitlab-ci.yml              # CI/CD pipeline
└── .env.example                # Environment template
```

## 🔄 DVC Workflow

```bash
# Pull data for a specific git tag
git checkout v1
docker compose run --rm trainer dvc pull

# Re-run training with exact same data
docker compose run --rm trainer python -m src.pipelines.zenml_pipeline \
    --config /app/configs/train_baseline.yaml

# Version new data changes
docker compose run --rm trainer dvc add data/coco128
docker compose run --rm trainer dvc push
git add data/coco128.dvc
git commit -m "Update dataset"
```

## 📊 Model Comparison

| Model | Config | Epochs | ImgSz | mAP50 | mAP50-95 |
|-------|--------|--------|-------|-------|----------|
| v1 (baseline) | train_baseline.yaml | 3 | 640 | ~0.45 | ~0.30 |
| v2 (improved) | train_v2.yaml | 5 | 416 | ~0.50 | ~0.33 |
| Optuna Best | optuna.yaml | var | var | ~0.52 | ~0.35 |

*Actual values depend on training run*

## 🔍 Reproducibility

Each MLflow run logs:
- `git_commit`: Git commit hash
- `dvc_data_version`: Data version hash
- `config_hash`: Configuration file hash
- `zenml_run_id`: ZenML pipeline run ID

To reproduce any experiment:
```bash
# Get reproducibility info from MLflow UI
git checkout <git_commit>
docker compose run --rm trainer dvc checkout
docker compose run --rm trainer python -m src.pipelines.zenml_pipeline \
    --config <original_config>
```

## 🧪 Testing

```bash
# Run all tests
docker compose run --rm trainer pytest tests/ -v

# Smoke test only (1 epoch)
docker compose run --rm trainer pytest tests/test_smoke_train.py -v

# API tests (requires running API)
docker compose up -d api
docker compose run --rm trainer pytest tests/test_api.py -v
```

## 📈 Optional: Monitoring Stack

```bash
# Start with monitoring
docker compose --profile monitoring up -d

# Access:
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
```

## 🏷️ Git Workflow

```bash
# Initial setup
git init
git add .
git commit -m "Initial MLOps setup"
git branch dev
git tag v1

# After v2 training
git checkout dev
# ... make changes ...
git commit -m "Improved model v2"
git checkout main
git merge dev
git tag v2
```

## 🛠️ Troubleshooting

### Services not starting
```bash
docker compose logs <service_name>
docker compose down -v  # Reset volumes
docker compose up -d
```

### MLflow connection issues
```bash
# Check MLflow is healthy
curl http://localhost:5000/health
docker compose logs mlflow
```

### MinIO bucket issues
```bash
# Recreate buckets
docker compose run --rm minio-init
```

### Model loading errors in API
```bash
# Check model exists in registry
curl http://localhost:5000/api/2.0/mlflow/registered-models/list
```

## 📝 License

MIT License - See LICENSE file for details.

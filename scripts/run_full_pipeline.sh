#!/bin/bash
# =============================================================================
# Full Pipeline Demo Script
# Runs the complete MLOps workflow from scratch
# =============================================================================

set -e

echo "=============================================="
echo "MLOps YOLO11: Full Pipeline Demo"
echo "=============================================="

# =============================================================================
# Step 1: Start Infrastructure
# =============================================================================
echo ""
echo ">>> Step 1: Starting infrastructure..."
docker compose up -d minio minio-init postgres mlflow
echo "Waiting for services to be ready..."
sleep 30

echo "Checking service health..."
docker compose ps
curl -sf http://localhost:5000/health || echo "MLflow not yet ready, waiting..."
sleep 10

# =============================================================================
# Step 2: Download and Version Dataset
# =============================================================================
echo ""
echo ">>> Step 2: Downloading and versioning dataset..."
docker compose run --rm trainer python -m src.data.ingest --setup-dvc

echo "Pushing data to MinIO..."
docker compose run --rm trainer dvc push || echo "DVC push skipped (no changes)"

# =============================================================================
# Step 3: Train Baseline (v1)
# =============================================================================
echo ""
echo ">>> Step 3: Training baseline model (v1)..."
docker compose run --rm trainer python -m src.pipelines.zenml_pipeline \
    --config /app/configs/train_baseline.yaml \
    --stage Production

# =============================================================================
# Step 4: Train Improved (v2)
# =============================================================================
echo ""
echo ">>> Step 4: Training improved model (v2)..."
docker compose run --rm trainer python -m src.pipelines.zenml_pipeline \
    --config /app/configs/train_v2.yaml

# =============================================================================
# Step 5: Deploy API
# =============================================================================
echo ""
echo ">>> Step 5: Deploying API with v1..."
MODEL_URI=models:/detector/1 docker compose up -d api
sleep 5

# Test API
echo "Testing API health..."
curl -sf http://localhost:8000/health

echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo ""
echo "Access Points:"
echo "  - MLflow UI: http://localhost:5000"
echo "  - MinIO Console: http://localhost:9001"
echo "  - API: http://localhost:8000"
echo ""
echo "Next Steps:"
echo "  - Run Optuna: docker compose run --rm trainer python -m src.train.optuna_tune --config /app/configs/optuna.yaml"
echo "  - Test API: curl -X POST http://localhost:8000/predict -F 'file=@image.jpg'"
echo "  - Demo rollback: ./scripts/demo_rollback.sh"

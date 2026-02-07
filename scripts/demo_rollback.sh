#!/bin/bash
# =============================================================================
# Demo Script: v1 → v2 Upgrade and Rollback
# =============================================================================

set -e

echo "=============================================="
echo "MLOps YOLO11 Demo: Model Upgrade & Rollback"
echo "=============================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test image (create a sample if not exists)
TEST_IMAGE="${1:-test_image.jpg}"

if [ ! -f "$TEST_IMAGE" ]; then
    echo -e "${BLUE}Creating sample test image...${NC}"
    docker compose run --rm trainer python -c "
from PIL import Image
import numpy as np
img = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
img.save('/app/data/test_image.jpg')
"
    TEST_IMAGE="data/test_image.jpg"
fi

# Function to test API
test_api() {
    local version=$1
    echo -e "\n${BLUE}Testing API with model version: $version${NC}"
    
    # Get model info
    echo "Model Info:"
    curl -s http://localhost:8000/model/info | python -m json.tool
    
    # Run prediction
    echo -e "\nPrediction Result:"
    curl -s -X POST http://localhost:8000/predict \
        -F "file=@$TEST_IMAGE" | python -m json.tool
}

# =============================================================================
# Step 1: Deploy v1
# =============================================================================
echo -e "\n${GREEN}=== Step 1: Deploying Model v1 ===${NC}"
docker compose down api 2>/dev/null || true
MODEL_URI=models:/detector/1 docker compose up -d api
sleep 5

echo -e "${GREEN}API deployed with v1${NC}"
test_api "v1"

echo -e "\n${BLUE}Press Enter to continue to v2 upgrade...${NC}"
read

# =============================================================================
# Step 2: Upgrade to v2
# =============================================================================
echo -e "\n${GREEN}=== Step 2: Upgrading to Model v2 ===${NC}"
docker compose down api
MODEL_URI=models:/detector/2 docker compose up -d api
sleep 5

echo -e "${GREEN}API upgraded to v2${NC}"
test_api "v2"

echo -e "\n${BLUE}Press Enter to rollback to v1...${NC}"
read

# =============================================================================
# Step 3: Rollback to v1
# =============================================================================
echo -e "\n${GREEN}=== Step 3: Rolling back to Model v1 ===${NC}"
docker compose down api
MODEL_URI=models:/detector/1 docker compose up -d api
sleep 5

echo -e "${GREEN}API rolled back to v1${NC}"
test_api "v1"

echo -e "\n${GREEN}=============================================="
echo "Demo Complete!"
echo "=============================================="
echo -e "${NC}"
echo "Summary:"
echo "  - Deployed v1: Baseline model"
echo "  - Upgraded to v2: Improved model"
echo "  - Rolled back to v1: Same predictions as initial"
echo ""
echo "Check MLflow UI for run comparisons: http://localhost:5000"

# =============================================================================
# Trainer Dockerfile
# Contains: DVC, ZenML, Optuna, Ultralytics YOLO, MLflow client
# =============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY requirements-trainer.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements-trainer.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY tests/ ./tests/

# Initialize DVC (will be configured at runtime)
RUN git config --global user.email "mlops@local" && \
    git config --global user.name "MLOps Trainer" && \
    git config --global init.defaultBranch main

# Default command
CMD ["bash"]

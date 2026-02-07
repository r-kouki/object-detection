# =============================================================================
# MLflow Server Dockerfile
# Includes psycopg2 for PostgreSQL backend and boto3 for S3
# =============================================================================
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    mlflow>=2.10.0 \
    psycopg2-binary \
    boto3

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]

"""
FastAPI Application for YOLO Object Detection API
Provides /predict endpoint for object detection inference.
"""

import io
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import mlflow
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from api.inference import YOLOInference, Detection


# =============================================================================
# Pydantic Models
# =============================================================================
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_uri: str | None


class DetectionResponse(BaseModel):
    success: bool
    detections: list[Detection]
    inference_time_ms: float
    model_uri: str


class ErrorResponse(BaseModel):
    success: bool
    error: str


# =============================================================================
# Global State
# =============================================================================
inference_engine: YOLOInference | None = None


# =============================================================================
# Lifespan Handler
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown."""
    global inference_engine
    
    # Startup
    print("=" * 60)
    print("Starting YOLO Object Detection API")
    print("=" * 60)
    
    # Setup MLflow
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI: {tracking_uri}")
    
    # Get model URI
    model_uri = os.environ.get("MODEL_URI", "models:/detector/1")
    print(f"Loading model from: {model_uri}")
    
    try:
        inference_engine = YOLOInference(model_uri=model_uri)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("API will start but /predict will return errors")
    
    print("=" * 60)
    
    yield
    
    # Shutdown
    print("Shutting down API...")


# =============================================================================
# FastAPI App
# =============================================================================
app = FastAPI(
    title="YOLO Object Detection API",
    description="API for object detection using YOLO11 models",
    version="1.0.0",
    lifespan=lifespan
)


# =============================================================================
# Endpoints
# =============================================================================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=inference_engine is not None and inference_engine.model is not None,
        model_uri=inference_engine.model_uri if inference_engine else None
    )


@app.post(
    "/predict",
    response_model=DetectionResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def predict(file: UploadFile = File(...)):
    """
    Perform object detection on an uploaded image.
    
    Args:
        file: Image file (jpg, jpeg, png, bmp, webp)
        
    Returns:
        Detection results with bounding boxes, scores, and classes
    """
    global inference_engine
    
    # Check if model is loaded
    if inference_engine is None or inference_engine.model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Check server logs."
        )
    
    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/bmp", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: {allowed_types}"
        )
    
    try:
        # Read image data
        image_data = await file.read()
        
        # Run inference
        start_time = time.time()
        detections = inference_engine.predict(image_data)
        inference_time = (time.time() - start_time) * 1000
        
        return DetectionResponse(
            success=True,
            detections=detections,
            inference_time_ms=round(inference_time, 2),
            model_uri=inference_engine.model_uri
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(e)}"
        )


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    if inference_engine is None:
        return {"error": "No model loaded"}
    
    return inference_engine.get_model_info()


# =============================================================================
# Prometheus Metrics (Optional)
# =============================================================================
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    from fastapi import Response
    
    # Metrics
    REQUEST_COUNT = Counter(
        "predict_requests_total",
        "Total prediction requests"
    )
    REQUEST_LATENCY = Histogram(
        "predict_request_latency_seconds",
        "Prediction request latency"
    )
    
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )

except ImportError:
    print("Prometheus client not installed - metrics disabled")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    
    uvicorn.run(app, host=host, port=port)

"""
YOLO Inference Utilities
Handles model loading, preprocessing, and postprocessing for inference.
"""

import io
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel
from ultralytics import YOLO


class Detection(BaseModel):
    """Represents a single detection."""
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2] normalized or pixel coords
    bbox_format: str = "xyxy"  # xyxy, xywh, etc.


class YOLOInference:
    """YOLO inference engine."""
    
    def __init__(
        self,
        model_uri: str | None = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 100,
        device: str = "cpu"
    ):
        """
        Initialize YOLO inference engine.
        
        Args:
            model_uri: Model URI (MLflow format or local path)
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            max_detections: Maximum detections per image
            device: Device to run inference on
        """
        self.model_uri = model_uri
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.device = device
        self.model: YOLO | None = None
        self.class_names: dict[int, str] = {}
        
        if model_uri:
            self.load_model(model_uri)
    
    def load_model(self, model_uri: str) -> None:
        """
        Load YOLO model from URI.
        
        Args:
            model_uri: Model URI (supports MLflow and local paths)
        """
        self.model_uri = model_uri
        
        # Determine model path
        if model_uri.startswith("models:/") or model_uri.startswith("runs:/"):
            # MLflow model URI
            model_path = self._load_from_mlflow(model_uri)
        else:
            # Local path
            model_path = model_uri
        
        # Load YOLO model
        print(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        
        # Get class names
        if hasattr(self.model, "names"):
            self.class_names = self.model.names
        
        print(f"Model loaded. Classes: {len(self.class_names)}")
    
    def _load_from_mlflow(self, model_uri: str) -> str:
        """
        Download model from MLflow and return local path.
        
        Args:
            model_uri: MLflow model URI
            
        Returns:
            Local path to downloaded model
        """
        import mlflow
        from mlflow.tracking import MlflowClient
        
        client = MlflowClient()
        
        if model_uri.startswith("models:/"):
            # Format: models:/name/version or models:/name/stage
            parts = model_uri.replace("models:/", "").split("/")
            model_name = parts[0]
            version_or_stage = parts[1] if len(parts) > 1 else "latest"
            
            # Get model version
            if version_or_stage.isdigit():
                version = version_or_stage
            else:
                # Get by stage
                versions = client.get_latest_versions(model_name, stages=[version_or_stage])
                if not versions:
                    versions = client.get_latest_versions(model_name)
                if not versions:
                    raise ValueError(f"No versions found for model: {model_name}")
                version = versions[0].version
            
            # Get model version details
            mv = client.get_model_version(model_name, version)
            artifact_uri = mv.source
            
        elif model_uri.startswith("runs:/"):
            # Format: runs:/run_id/path
            artifact_uri = model_uri
        else:
            raise ValueError(f"Unsupported MLflow URI format: {model_uri}")
        
        # Download artifact
        local_path = mlflow.artifacts.download_artifacts(artifact_uri)
        
        # Find .pt file
        local_path = Path(local_path)
        if local_path.is_dir():
            pt_files = list(local_path.glob("*.pt"))
            if pt_files:
                return str(pt_files[0])
            # Check subdirectories
            pt_files = list(local_path.glob("**/*.pt"))
            if pt_files:
                return str(pt_files[0])
        
        return str(local_path)
    
    def preprocess(self, image_data: bytes) -> Image.Image:
        """
        Preprocess image data.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            PIL Image
        """
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return image
    
    def predict(self, image_data: bytes) -> list[Detection]:
        """
        Run inference on image.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            List of Detection objects
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Preprocess
        image = self.preprocess(image_data)
        
        # Run inference
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            device=self.device,
            verbose=False
        )
        
        # Parse results
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            
            boxes = result.boxes
            for i in range(len(boxes)):
                # Get box coordinates
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                
                detections.append(Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=round(conf, 4),
                    bbox=[round(x, 2) for x in xyxy],
                    bbox_format="xyxy"
                ))
        
        return detections
    
    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        return {
            "model_uri": self.model_uri,
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "device": self.device,
        }


# =============================================================================
# Utility Functions
# =============================================================================
def draw_detections(
    image: Image.Image,
    detections: list[Detection],
    line_width: int = 2,
    font_size: int = 12
) -> Image.Image:
    """
    Draw detection boxes on image.
    
    Args:
        image: PIL Image
        detections: List of detections
        line_width: Box line width
        font_size: Label font size
        
    Returns:
        Image with drawn detections
    """
    from PIL import ImageDraw, ImageFont
    
    draw = ImageDraw.Draw(image)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    
    # Color palette
    colors = [
        "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF",
        "#00FFFF", "#FFA500", "#800080", "#008000", "#000080"
    ]
    
    for det in detections:
        color = colors[det.class_id % len(colors)]
        
        # Draw box
        x1, y1, x2, y2 = det.bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        
        # Draw label
        label = f"{det.class_name} {det.confidence:.2f}"
        bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=color)
        draw.text((x1, y1), label, fill="white", font=font)
    
    return image

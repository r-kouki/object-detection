"""
Smoke Test for YOLO Training
Runs a minimal training (1 epoch) to verify setup.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest


def test_ultralytics_import():
    """Test that Ultralytics can be imported."""
    from ultralytics import YOLO
    assert YOLO is not None


def test_model_download():
    """Test that YOLO11n model can be downloaded."""
    from ultralytics import YOLO
    
    model = YOLO("yolo11n.pt")
    assert model is not None


def test_smoke_train():
    """
    Smoke test: Train for 1 epoch with minimal settings.
    This verifies the entire training pipeline works.
    """
    from ultralytics import YOLO
    
    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        model = YOLO("yolo11n.pt")
        
        # Train with minimal settings
        results = model.train(
            data="coco128.yaml",  # Will download if not present
            epochs=1,
            imgsz=64,  # Tiny images for speed
            batch=4,
            device="cpu",
            project=tmpdir,
            name="smoke_test",
            exist_ok=True,
            verbose=False,
        )
        
        # Check that training completed
        assert results is not None
        
        # Check that weights were saved
        weights_path = Path(tmpdir) / "smoke_test" / "weights" / "best.pt"
        # Note: best.pt might not exist for 1 epoch, check last.pt
        last_weights = Path(tmpdir) / "smoke_test" / "weights" / "last.pt"
        assert weights_path.exists() or last_weights.exists()


def test_model_validation():
    """Test model validation works."""
    from ultralytics import YOLO
    
    model = YOLO("yolo11n.pt")
    
    # Run quick validation
    with tempfile.TemporaryDirectory() as tmpdir:
        results = model.val(
            data="coco128.yaml",
            imgsz=64,
            batch=4,
            device="cpu",
            project=tmpdir,
            name="val_test",
            verbose=False,
        )
        
        assert results is not None
        assert hasattr(results, "box")


def test_model_predict():
    """Test model inference works."""
    from ultralytics import YOLO
    import numpy as np
    from PIL import Image
    
    model = YOLO("yolo11n.pt")
    
    # Create a dummy image
    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    )
    
    # Run prediction
    results = model(dummy_image, device="cpu", verbose=False)
    
    assert results is not None
    assert len(results) == 1


@pytest.mark.skipif(
    os.environ.get("MLFLOW_TRACKING_URI") is None,
    reason="MLflow not configured"
)
def test_mlflow_connection():
    """Test MLflow connection works."""
    import mlflow
    
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Try to list experiments
    experiments = mlflow.search_experiments()
    assert experiments is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

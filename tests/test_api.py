"""
API Tests
Tests for the FastAPI inference endpoint.
"""

import io
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np
from PIL import Image


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_test_image() -> bytes:
    """Create a test image as bytes."""
    # Create a random image
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Convert to bytes
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    
    return buffer.getvalue()


class TestAPIEndpoints:
    """Tests for API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client with mocked model."""
        # Mock the inference engine before importing app
        with patch("api.main.YOLOInference") as MockInference:
            # Setup mock
            mock_engine = MagicMock()
            mock_engine.model = MagicMock()
            mock_engine.model_uri = "test://model"
            mock_engine.predict.return_value = [
                {
                    "class_id": 0,
                    "class_name": "person",
                    "confidence": 0.95,
                    "bbox": [10.0, 10.0, 50.0, 50.0],
                    "bbox_format": "xyxy"
                }
            ]
            mock_engine.get_model_info.return_value = {
                "model_uri": "test://model",
                "num_classes": 80
            }
            MockInference.return_value = mock_engine
            
            from fastapi.testclient import TestClient
            from api.main import app
            
            # Override the inference engine
            import api.main
            api.main.inference_engine = mock_engine
            
            yield TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
    
    def test_predict_endpoint(self, client):
        """Test prediction endpoint with valid image."""
        image_data = create_test_image()
        
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", image_data, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "detections" in data
        assert "inference_time_ms" in data
    
    def test_predict_invalid_file_type(self, client):
        """Test prediction with invalid file type."""
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        
        assert response.status_code == 400
    
    def test_model_info_endpoint(self, client):
        """Test model info endpoint."""
        response = client.get("/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "model_uri" in data


class TestInference:
    """Tests for inference utilities."""
    
    def test_detection_model(self):
        """Test Detection Pydantic model."""
        from api.inference import Detection
        
        det = Detection(
            class_id=0,
            class_name="person",
            confidence=0.95,
            bbox=[10.0, 10.0, 50.0, 50.0]
        )
        
        assert det.class_id == 0
        assert det.class_name == "person"
        assert det.confidence == 0.95
        assert len(det.bbox) == 4
    
    def test_preprocess(self):
        """Test image preprocessing."""
        from api.inference import YOLOInference
        
        engine = YOLOInference.__new__(YOLOInference)
        engine.model = None
        
        image_data = create_test_image()
        image = engine.preprocess(image_data)
        
        assert image is not None
        assert image.mode == "RGB"


@pytest.mark.skipif(
    os.environ.get("API_URL") is None,
    reason="API_URL not set for integration tests"
)
class TestIntegration:
    """Integration tests against running API."""
    
    def test_live_health(self):
        """Test health endpoint on live API."""
        import requests
        
        api_url = os.environ.get("API_URL", "http://localhost:8000")
        response = requests.get(f"{api_url}/health")
        
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_live_predict(self):
        """Test predict endpoint on live API."""
        import requests
        
        api_url = os.environ.get("API_URL", "http://localhost:8000")
        image_data = create_test_image()
        
        response = requests.post(
            f"{api_url}/predict",
            files={"file": ("test.jpg", image_data, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

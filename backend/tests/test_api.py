"""
Test Suite — PlantMD Pro Backend
Run: pytest tests/ -v --cov=. --cov-report=term-missing
"""
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def client():
    """Create FastAPI test client."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from main import app
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Generate a valid 224x224 RGB JPEG image."""
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def tiny_image_bytes():
    """Generate a too-small image (32x32)."""
    img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def mock_model():
    """Mock model returning a healthy prediction."""
    mock = MagicMock()
    mock._is_loaded = True
    mock.predict.return_value = {
        "class_key": "healthy",
        "display_name": "Healthy",
        "confidence": 0.97,
        "is_healthy": True,
        "severity": "NONE",
        "affected_crop": "N/A",
        "treatment": {
            "immediate_action": "No action needed.",
            "fungicide": "None required.",
            "prevention": "Continue good agricultural practices.",
        },
        "top3": [],
        "demo_mode": False,
    }
    mock.generate_gradcam.return_value = None
    return mock


# ── Health Endpoint Tests ─────────────────────────────────────────────────────
class TestHealthEndpoints:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_required_fields(self, client):
        r = client.get("/health")
        data = r.json()
        assert "status" in data
        assert "version" in data
        assert "model_loaded" in data

    def test_liveness_returns_alive(self, client):
        r = client.get("/health/live")
        assert r.status_code == 200
        assert r.json()["status"] == "alive"

    def test_root_endpoint(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "service" in r.json()


# ── Prediction Endpoint Tests ─────────────────────────────────────────────────
class TestPredictEndpoint:
    def test_predict_healthy_image(self, client, sample_image_bytes, mock_model):
        with patch("api.routes.predict.get_model", return_value=mock_model):
            r = client.post(
                "/api/v1/predict",
                files={"file": ("leaf.jpg", sample_image_bytes, "image/jpeg")},
                data={"generate_heatmap": "false"},
            )
        assert r.status_code == 200
        data = r.json()
        assert data["display_name"] == "Healthy"
        assert data["confidence"] == 0.97
        assert data["is_healthy"] is True

    def test_predict_returns_treatment(self, client, sample_image_bytes):
        mock = MagicMock()
        mock._is_loaded = True
        mock.predict.return_value = {
            "class_key": "Tomato___Late_blight",
            "display_name": "Tomato Late Blight",
            "confidence": 0.93,
            "is_healthy": False,
            "severity": "HIGH",
            "affected_crop": "Tomato",
            "treatment": {
                "immediate_action": "Remove infected parts.",
                "fungicide": "Chlorothalonil",
                "prevention": "Improve airflow.",
            },
            "top3": [],
            "demo_mode": False,
        }
        mock.generate_gradcam.return_value = None

        with patch("api.routes.predict.get_model", return_value=mock):
            r = client.post(
                "/api/v1/predict",
                files={"file": ("leaf.jpg", sample_image_bytes, "image/jpeg")},
                data={"generate_heatmap": "false"},
            )
        assert r.status_code == 200
        data = r.json()
        assert data["severity"] == "HIGH"
        assert "treatment" in data
        assert data["treatment"]["fungicide"] == "Chlorothalonil"

    def test_predict_low_confidence_returns_error(self, client, sample_image_bytes):
        mock = MagicMock()
        mock._is_loaded = True
        mock.predict.return_value = {
            "class_key": "healthy",
            "display_name": "Healthy",
            "confidence": 0.1,  # Below threshold
            "is_healthy": True,
            "severity": "NONE",
            "affected_crop": "N/A",
            "treatment": {},
            "top3": [],
        }
        with patch("api.routes.predict.get_model", return_value=mock):
            r = client.post(
                "/api/v1/predict",
                files={"file": ("leaf.jpg", sample_image_bytes, "image/jpeg")},
                data={"generate_heatmap": "false"},
            )
        assert r.status_code == 200  # LowConfidenceError returns 200 with error
        assert r.json().get("error") == "LOW_CONFIDENCE"

    def test_predict_no_file_returns_422(self, client):
        r = client.post("/api/v1/predict")
        assert r.status_code == 422

    def test_predict_classes_endpoint(self, client):
        r = client.get("/api/v1/predict/classes")
        assert r.status_code == 200
        data = r.json()
        assert "total" in data
        assert "classes" in data
        assert data["total"] > 0


# ── Image Validation Tests ─────────────────────────────────────────────────────
class TestImageValidation:
    def test_valid_jpeg_passes(self, sample_image_bytes):
        from ml.preprocessor import validate_and_load_image
        arr, hash_ = validate_and_load_image(sample_image_bytes, "test.jpg", "image/jpeg")
        assert arr.shape == (224, 224, 3)
        assert len(hash_) == 64

    def test_oversized_file_raises(self):
        from core.exceptions import ImageTooLargeError
        from ml.preprocessor import validate_and_load_image

        # 11MB of fake data
        huge = b"\xff\xd8\xff" + b"A" * (11 * 1024 * 1024)
        with pytest.raises(ImageTooLargeError):
            validate_and_load_image(huge, "big.jpg")

    def test_unsupported_extension_raises(self, sample_image_bytes):
        from core.exceptions import UnsupportedFormatError
        from ml.preprocessor import validate_and_load_image

        with pytest.raises(UnsupportedFormatError):
            validate_and_load_image(sample_image_bytes, "file.bmp")

    def test_tiny_image_raises(self, tiny_image_bytes):
        from core.exceptions import InvalidImageError
        from ml.preprocessor import validate_and_load_image

        with pytest.raises(InvalidImageError):
            validate_and_load_image(tiny_image_bytes, "small.jpg")

    def test_corrupted_file_raises(self):
        from core.exceptions import InvalidImageError
        from ml.preprocessor import validate_and_load_image

        with pytest.raises(InvalidImageError):
            validate_and_load_image(b"\xff\xd8\xff" + b"garbage" * 100, "bad.jpg")


# ── Auth Tests ─────────────────────────────────────────────────────────────────
class TestAuth:
    def test_login_valid_credentials(self, client):
        r = client.post(
            "/api/v1/auth/token",
            data={"username": "demo", "password": "demo"},
        )
        assert r.status_code == 200
        assert "access_token" in r.json()

    def test_login_invalid_credentials(self, client):
        r = client.post(
            "/api/v1/auth/token",
            data={"username": "nobody", "password": "wrong"},
        )
        assert r.status_code == 401

    def test_get_me_requires_token(self, client):
        r = client.get("/api/v1/auth/me")
        assert r.status_code == 401

    def test_get_me_with_valid_token(self, client):
        login = client.post(
            "/api/v1/auth/token",
            data={"username": "demo", "password": "demo"},
        )
        token = login.json()["access_token"]
        r = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert r.json()["username"] == "demo"


# ── Model Tests ────────────────────────────────────────────────────────────────
class TestModel:
    def test_model_demo_prediction_structure(self):
        from ml.model import PlantDiseaseModel
        m = PlantDiseaseModel()
        m._is_loaded = True
        m._labels = list(__import__("ml.model", fromlist=["DISEASE_METADATA"]).DISEASE_METADATA.keys())
        result = m._demo_prediction()
        assert "display_name" in result
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1
        assert "treatment" in result

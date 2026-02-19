"""
Tests for the FastAPI inference endpoint.
Uses TestClient with mocked model, encoders, and class mapping
to avoid loading real ML models during CI.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

import multimodal_ai.api.main as api_module
from multimodal_ai.api.main import app


# Helpers
def _make_jpeg_bytes() -> bytes:
    """Generate a minimal valid JPEG image as bytes."""
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(128, 64, 32)).save(buf, format="JPEG")
    return buf.getvalue()


def _make_mock_model(num_classes: int = 27) -> MagicMock:
    """Return a mock nn.Module that outputs uniform logits."""
    mock = MagicMock()
    mock.return_value = torch.zeros(1, num_classes)
    return mock


# Fixtures
@pytest.fixture()
def client():
    """
    TestClient with all ML components mocked at module level.
    Avoids loading real models (MLflow, sentence-transformers, timm).
    """
    idx_to_label = {str(i): f"category_{i}" for i in range(27)}

    with (
        patch.object(api_module, "_model", _make_mock_model(27)),
        patch.object(
            api_module,
            "_text_encoder",
            MagicMock(encode_text_infer=lambda text: np.zeros(768, dtype=np.float32)),
        ),
        patch.object(
            api_module,
            "_image_encoder",
            MagicMock(encode_image_bytes=lambda b: np.zeros(384, dtype=np.float32)),
        ),
        patch.object(api_module, "_idx_to_label", idx_to_label),
    ):
        yield TestClient(app)


# Tests
def test_health_ok(client):
    """Health endpoint returns 200 when model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_no_model(client):
    """Health endpoint returns 503 when model is not loaded."""
    with patch.object(api_module, "_model", None):
        response = client.get("/health")
    assert response.status_code == 503


def test_predict_returns_valid_response(client):
    """Predict endpoint returns expected fields with valid input."""
    response = client.post(
        "/predict",
        data={
            "designation": "Livre de cuisine italienne",
            "description": "Recettes du terroir",
        },
        files={"image": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert "predicted_class_index" in body
    assert "predicted_label" in body
    assert "confidence" in body
    assert "top5" in body


def test_predict_confidence_is_valid(client):
    """Confidence score is between 0 and 1."""
    response = client.post(
        "/predict",
        data={"designation": "Jeu de société", "description": "Pour toute la famille"},
        files={"image": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    confidence = response.json()["confidence"]
    assert 0.0 <= confidence <= 1.0


def test_predict_top5_length(client):
    """Top-5 contains at most 5 entries."""
    response = client.post(
        "/predict",
        data={"designation": "Smartphone Android"},
        files={"image": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    top5 = response.json()["top5"]
    assert 1 <= len(top5) <= 5


def test_predict_top5_fields(client):
    """Each top-5 entry contains class_index, label, and confidence."""
    response = client.post(
        "/predict",
        data={
            "designation": "Casque audio Bluetooth",
            "description": "Son haute fidélité",
        },
        files={"image": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    for entry in response.json()["top5"]:
        assert "class_index" in entry
        assert "label" in entry
        assert "confidence" in entry


def test_predict_label_in_mapping(client):
    """Predicted label exists in the class mapping."""
    idx_to_label = {str(i): f"category_{i}" for i in range(27)}
    response = client.post(
        "/predict",
        data={"designation": "Vélo de route carbone"},
        files={"image": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    label = response.json()["predicted_label"]
    assert label in idx_to_label.values()


def test_predict_description_optional(client):
    """Predict endpoint returns 200 when description is omitted."""
    response = client.post(
        "/predict",
        data={"designation": "Produit sans description"},
        files={"image": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200


def test_predict_missing_designation(client):
    """Predict endpoint returns 422 when designation field is missing."""
    response = client.post(
        "/predict",
        files={"image": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 422


def test_predict_missing_image(client):
    """Predict endpoint returns 422 when image field is missing."""
    response = client.post(
        "/predict",
        data={"designation": "Produit sans image"},
    )
    assert response.status_code == 422


def test_predict_model_not_loaded(client):
    """Predict endpoint returns 503 when model is not loaded."""
    with patch.object(api_module, "_model", None):
        response = client.post(
            "/predict",
            data={"designation": "Test"},
            files={"image": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
    assert response.status_code == 503

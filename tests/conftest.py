from __future__ import annotations

import pytest

from multimodal_ai.features.base_image_embedder import BaseImageEmbedder
from multimodal_ai.features.base_text_embedder import BaseTextEmbedder
from multimodal_ai.features.image_encoder_infer import ImageEncoderInfer
from multimodal_ai.features.image_encoder_train import ImageEncoderTrain
from multimodal_ai.features.text_encoder_infer import TextEncoderInfer
from multimodal_ai.features.text_encoder_train import TextEncoderTrain


@pytest.fixture(scope="session")
def image_embedder() -> BaseImageEmbedder:
    """Session-scoped BaseImageEmbedder (ResNet50/CPU) for integration tests."""

    print("\n[SETUP] Loading BaseImageEmbedder (resnet50/CPU)...")
    return BaseImageEmbedder(
        model_name="resnet50",
        device="cpu",
        normalize_embeddings=True,
        batch_size=3,
    )


@pytest.fixture(scope="session")
def text_embedder() -> BaseTextEmbedder:
    """Session-scoped BaseTextEmbedder (all-MiniLM-L6-v2/CPU) for integration tests."""

    print("\n[SETUP] Loading BaseTextEmbedder (all-MiniLM-L6-v2/CPU)...")
    return BaseTextEmbedder(
        model_name="all-MiniLM-L6-v2",
        device="cpu",
        normalize_embeddings=True,
        batch_size=3,
    )


@pytest.fixture(scope="session")
def infer_encoder() -> ImageEncoderInfer:
    """Inference encoder (ResNet18/CPU) for single-image bytes testing."""

    print("\n[SETUP] Loading ImageEncoderInfer (ResNet18/CPU)...")
    return ImageEncoderInfer(model_name="resnet18", device="cpu")


@pytest.fixture(scope="session")
def train_encoder() -> ImageEncoderTrain:
    """Training encoder (ResNet18/CPU) with batch_size=2 to test pagination logic."""

    print("\n[SETUP] Loading ImageEncoderTrain (ResNet18/CPU)...")
    return ImageEncoderTrain(model_name="resnet18", device="cpu", batch_size=2)


@pytest.fixture(scope="session")
def infer_text_encoder() -> TextEncoderInfer:
    """Inference text encoder (all-MiniLM-L6-v2) for API logic simulation."""

    print("\n[SETUP] Loading TextEncoderInfer (all-MiniLM-L6-v2/CPU)...")
    return TextEncoderInfer(model_name="all-MiniLM-L6-v2", device="cpu")


@pytest.fixture(scope="session")
def train_text_encoder() -> TextEncoderTrain:
    """Training text encoder (all-MiniLM-L6-v2) with batch_size=2 for bulk processing."""

    print("\n[SETUP] Loading TextEncoderTrain (all-MiniLM-L6-v2/CPU)...")
    return TextEncoderTrain(model_name="all-MiniLM-L6-v2", device="cpu", batch_size=2)

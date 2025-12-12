from __future__ import annotations

import numpy as np
import torch

from multimodal_ai.config.settings import settings
from multimodal_ai.features.base_image_embedder import BaseImageEmbedder


def test_fixture_initialization(image_embedder):
    """Verifies that the shared fixture is correctly configured (CPU, batch size, normalization)."""
    assert image_embedder.device == "cpu"
    assert image_embedder.batch_size == 3
    assert image_embedder.normalize_embeddings is True


def test_default_initialization():
    """Ensures fallback to global settings (config.settings) when no args are provided."""
    embedder = BaseImageEmbedder()
    assert embedder.device == settings.DEFAULT_DEVICE
    assert embedder.model_name == settings.IMAGE_MODEL_NAME
    assert embedder.normalize_embeddings == settings.IMAGE_NORMALIZE
    assert embedder.batch_size == settings.get_batch_size(embedder.device)
    assert embedder.model is not None


def test_encode_tensor_batch(image_embedder):
    """Validates output shape, data type, numerical stability, and normalization."""
    input_size = image_embedder.model_cfg["input_size"]
    H, W = input_size[1], input_size[2]
    batch_size = 2
    image_batch = torch.randn(batch_size, 3, H, W)

    embeddings = image_embedder.encode_tensor_batch(image_batch)

    # Shape & Type
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (
        batch_size,
        image_embedder.get_embedding_dim(),
    )

    # Stability (No NaNs/Infs)
    assert not np.isnan(embeddings).any()
    assert not np.isinf(embeddings).any()

    # Normalization (L2 Norm = 1.0)
    if image_embedder.normalize_embeddings:
        norms = np.linalg.norm(embeddings, ord=2, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)


def test_get_embedding_dim(image_embedder):
    """Ensures the returned dimension matches the underlying timm model features."""
    assert isinstance(image_embedder.get_embedding_dim(), int)
    assert image_embedder.get_embedding_dim() == image_embedder.model.num_features

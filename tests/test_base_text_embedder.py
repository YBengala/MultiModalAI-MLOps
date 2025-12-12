from __future__ import annotations

import numpy as np

from multimodal_ai.config.settings import settings
from multimodal_ai.features.base_text_embedder import BaseTextEmbedder


def test_fixture_configuration(text_embedder: BaseTextEmbedder):
    """Verifies the shared fixture configuration (CPU, normalization, batch size)."""
    assert text_embedder.device == "cpu"
    assert text_embedder.normalize_sentence is True
    assert text_embedder.batch_size == 3
    assert text_embedder.model is not None


def test_default_initialization():
    """Ensures fallback to global settings (config.settings) when no args are provided."""
    embedder = BaseTextEmbedder()
    assert embedder.device == settings.DEFAULT_DEVICE
    assert embedder.model_name == settings.TEXT_MODEL_NAME
    assert embedder.normalize_sentence == settings.TEXT_NORMALIZE


def test_get_embedding_dim(text_embedder: BaseTextEmbedder):
    """Ensures the returned dimension matches the underlying SentenceTransformer model."""
    # Test type
    assert isinstance(text_embedder.get_embedding_dim(), int)

    # Test features dimensions
    assert (
        text_embedder.get_embedding_dim()
        == text_embedder.model.get_sentence_embedding_dimension()
    )


def test_encode_text_single_string(text_embedder: BaseTextEmbedder):
    """Validates output shape, data type, and numerical stability for single input."""
    text = "histoire du monde"
    embedding = text_embedder.encode_text(text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (1, text_embedder.get_embedding_dim())
    assert not np.isnan(embedding).any()


def test_encode_text_batch_list(text_embedder: BaseTextEmbedder):
    """Verifies that the output shape matches (Batch_Size, Dimension) for list inputs."""
    texts = ["ref:1445855 piscine <br>", "voiture jouet", "livre 14552329666555"]
    embeddings = text_embedder.encode_text(texts)

    expected_dim = text_embedder.get_embedding_dim()
    assert embeddings.shape == (3, expected_dim)


def test_normalization_logic(text_embedder: BaseTextEmbedder):
    """Ensures embeddings are unit-length L2-normalized (norm approx 1.0)."""
    text = "fifa 2025 <br>"
    embedding = text_embedder.encode_text(text)

    norm = np.linalg.norm(embedding, ord=2, axis=1)
    assert np.allclose(norm, 1.0, atol=1e-5)

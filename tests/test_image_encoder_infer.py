from __future__ import annotations

import io

import numpy as np
from PIL import Image

from multimodal_ai.features.image_encoder_infer import ImageEncoderInfer


def test_encode_valid_image_bytes(infer_encoder: ImageEncoderInfer):
    """Validates the full inference pipeline (Bytes -> Embedding) with valid PNG data."""
    dummy_img = Image.new("RGB", (64, 64), color="blue")
    buffer = io.BytesIO()
    dummy_img.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    embedding = infer_encoder.encode_image_bytes(image_bytes)

    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert embedding.shape[0] == infer_encoder.get_embedding_dim()
    assert not np.isnan(embedding).any()


def test_encode_corrupted_image(infer_encoder: ImageEncoderInfer):
    """Ensures corrupted bytes trigger the fallback mechanism (white placeholder) without crashing."""
    inv_img = b"je suis une image invalide"

    embedding = infer_encoder.encode_image_bytes(inv_img)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] == infer_encoder.get_embedding_dim()
    assert not np.isnan(embedding).any()

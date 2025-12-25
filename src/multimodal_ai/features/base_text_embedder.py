"""
Text Feature Extractor :
    - Base class used for both `TextEncoderTrain` and `TextEncoderInfer`.
    - Input : List of strings (texts).
    - Output : (N, D) numpy embeddings L2-normalized.
"""

from __future__ import annotations

from typing import Iterable, cast

import numpy as np
from sentence_transformers import SentenceTransformer

from multimodal_ai.config.settings import settings


class BaseTextEmbedder:
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        normalize_embeddings: bool | None = None,
    ) -> None:
        # Initialize configuration
        self.model_name = model_name or settings.TEXT_MODEL_NAME
        self.device = device or settings.DEFAULT_DEVICE
        self.batch_size = batch_size or settings.get_batch_size(self.device)
        self.normalize_sentence = (
            normalize_embeddings
            if normalize_embeddings is not None
            else settings.TEXT_NORMALIZE
        )

        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.model.eval()

    def encode_text(
        self,
        text: str | Iterable[str],
    ) -> np.ndarray:
        texts = [text] if isinstance(text, str) else list(text)
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize_sentence,
            convert_to_numpy=True,
        )
        return embeddings

    def get_embedding_dim(self) -> int:
        return cast(int, self.model.get_sentence_embedding_dimension())

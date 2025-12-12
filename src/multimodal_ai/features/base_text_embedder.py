from __future__ import annotations

from typing import Iterable, cast

import numpy as np
from sentence_transformers import SentenceTransformer

from multimodal_ai.config.settings import settings


class BaseTextEmbedder:
    """Provides text embedding extraction using SentenceTransformer backbones.

    Loads the configured model on the selected device, manages batching, and
    optionally L2-normalizes output embeddings for cosine-similarity workflows.
    """

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
        """Encode one or more texts into NumPy embeddings.

        Args:
            text: Single string or iterable of strings to embed.

        Returns:
            np.ndarray: Array shaped (N, D) where N is the number of inputs and
            D is the model output dimension.L2-normalized when `normalize_embeddings` is True.
        """

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
        """Return the embedding dimension reported by the loaded model."""

        return cast(int, self.model.get_sentence_embedding_dimension())

from __future__ import annotations

import numpy as np

from multimodal_ai.features.base_text_embedder import BaseTextEmbedder
from multimodal_ai.features.text_cleaner import input_text_infer


class TextEncoderInfer(BaseTextEmbedder):
    """Inference text encoder for API usage.

    Combines text cleaning (designation + description) and embedding extraction
    into a single pipeline step.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Initializes the inference encoder.

        Args:
            model_name: SentenceTransformer model name.
            device: 'cpu' or 'cuda'.
            batch_size: Inference batch size.
        """

        super().__init__(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            normalize_embeddings=True,
        )

    def encode_text_infer(
        self,
        designation: str,
        description: str | None = None,
    ) -> np.ndarray:
        """Cleans and encodes product text data into a single vector.

        Merges designation and description, applies cleaning rules (HTML, regex),
        and generates the embedding.

        Args:
            designation: Main product title or name.
            description: Detailed product description (optional).

        Returns:
            np.ndarray: 1D embedding vector.
        """

        text_input = input_text_infer(designation, description)
        text_infer_feats = self.encode_text(text_input)

        return text_infer_feats[0]

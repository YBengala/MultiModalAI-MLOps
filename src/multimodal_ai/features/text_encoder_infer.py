"""
Inference Text Encoder :
    - Single Product Processing.
    - Cleaning (HTML/Regex) -> Merging -> Encoding.
    - Input : Designation (str) + Description (str).
    - Output : 1D Numpy Array (Embedding vector).
"""

from __future__ import annotations

import numpy as np

from multimodal_ai.features.base_text_embedder import BaseTextEmbedder
from multimodal_ai.features.text_cleaner import input_text_infer


class TextEncoderInfer(BaseTextEmbedder):
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            normalize_embeddings=True,
        )

    def encode_text_infer(
        self, designation: str, description: str | None = None
    ) -> np.ndarray:
        return self.encode_text(input_text_infer(designation, description))[0]

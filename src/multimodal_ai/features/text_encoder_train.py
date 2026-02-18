"""
Batch Text Encoder (Training) :
    - Input : list of pre-cleaned texts.
    - Output : (N, D) Numpy Array.
"""

from __future__ import annotations

import numpy as np

from multimodal_ai.features.base_text_embedder import BaseTextEmbedder


class TextEncoderTrain(BaseTextEmbedder):
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            batch_size=batch_size,
            device=device,
            normalize_embeddings=True,
        )

    def text_train_encodings(self, df, col_text="text") -> np.ndarray:
        if df.empty:
            return np.empty((0, self.get_embedding_dim()), dtype=np.float32)
        return self.encode_text(df[col_text].tolist())

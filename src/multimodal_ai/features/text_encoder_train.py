from __future__ import annotations

import numpy as np
import pandas as pd

from multimodal_ai.features.base_text_embedder import BaseTextEmbedder
from multimodal_ai.features.text_cleaner import input_text_train


class TextEncoderTrain(BaseTextEmbedder):
    """Batch text encoder for training pipelines.

    Processes DataFrames to generate normalized text embeddings suitable for
    training or building search indices.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Initializes the training text encoder.

        Args:
            model_name: SentenceTransformer model name.
            device: 'cpu' or 'cuda'.
            batch_size: Batch size for bulk encoding.
        """
        super().__init__(
            model_name=model_name,
            batch_size=batch_size,
            device=device,
            normalize_embeddings=True,
        )

    def text_train_encodings(
        self,
        df: pd.DataFrame,
        col_designation: str = "designation",
        col_description: str = "description",
    ) -> np.ndarray:
        """Generates embeddings for all rows in the DataFrame.

        Applies cleaning and concatenation of designation and description
        before encoding.

        Args:
            df: Input DataFrame containing text columns.
            col_designation: Column name for product titles.
            col_description: Column name for product descriptions.

        Returns:
            np.ndarray: Matrix of shape (N, D) containing normalized embeddings.
        """

        if df.empty:
            dim = self.get_embedding_dim()
            return np.empty((0, dim), dtype=np.float32)

        df_clean = input_text_train(
            df, col_des=col_designation, col_desc=col_description
        )

        texts = df_clean["input_text"].tolist()
        text_train_feats = self.encode_text(texts)

        return text_train_feats

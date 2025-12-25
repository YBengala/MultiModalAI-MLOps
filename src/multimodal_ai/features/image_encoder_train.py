"""
Batch Image Encoder (Training) :
    - Input : DataFrame with Image file paths column.
    - Output : (N, D) Numpy array.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from multimodal_ai.features.base_image_embedder import BaseImageEmbedder
from multimodal_ai.features.image_dataset import ImageDataset


class ImageEncoderTrain(BaseImageEmbedder):
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        num_workers: int = 0,
    ) -> None:
        super().__init__(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            normalize_embeddings=True,
        )
        self.num_workers = num_workers

    def image_train_encodings(
        self,
        df: pd.DataFrame,
        path_column: str,
    ) -> np.ndarray:
        dataset = ImageDataset(df, path_column=path_column, transform=self.transform)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device == "cuda"),
        )

        img_encodings = []

        with torch.no_grad():
            for batch in loader:
                feats = self.encode_tensor_batch(batch)
                img_encodings.append(feats)

        if len(img_encodings) == 0:
            dim = self.get_embedding_dim()
            return np.empty((0, dim), dtype=np.float32)

        return np.vstack(img_encodings)

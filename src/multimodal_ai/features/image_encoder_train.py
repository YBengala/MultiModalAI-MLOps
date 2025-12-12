from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from multimodal_ai.features.base_image_embedder import BaseImageEmbedder
from multimodal_ai.features.image_dataset import ImageDataset


class ImageEncoderTrain(BaseImageEmbedder):
    """Batch image encoder for training pipelines.

    Extracts normalized embeddings from a DataFrame of image paths using
    multi-threaded batch processing.

    Attributes:
        num_workers (int): Number of subprocesses for data loading.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        num_workers: int = 0,
    ) -> None:
        """Initializes the encoder.

        Args:
            model_name: Timm model name (default: settings.IMAGE_MODEL_NAME).
            device: 'cpu' or 'cuda' (default: settings.DEFAULT_DEVICE).
            batch_size: Images per batch (default: dynamic based on device).
            num_workers: Subprocesses for data loading (default: 0).
        """

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
        """Generates embeddings for images listed in a DataFrame.

        Args:
            df: DataFrame containing image paths.
            path_column: Column name containing file paths.

        Returns:
            np.ndarray: Embeddings array of shape (N, D). Returns empty (0, D)
            if input is empty.
        """

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

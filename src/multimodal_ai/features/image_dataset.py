"""
Image Loading Dataset (PyTorch) :
    - Input : DataFrame + column name containing image file paths.
    - Output : Transformed Tensor (C, H, W).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from multimodal_ai.config.settings import settings


class ImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        path_column: str,
        transform: Any | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.image_paths: list[str] = df[path_column].astype(str).tolist()
        self.transform = transform
        self.image_size = settings.IMAGE_SIZE

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Any:
        path = self.image_paths[index].strip()
        try:
            image = Image.open(path).convert("RGB")

        except Exception:
            # white image if error in loading
            image = Image.new(
                "RGB", (self.image_size, self.image_size), (255, 255, 255)
            )

        if self.transform is not None:
            image = self.transform(image)

        return image

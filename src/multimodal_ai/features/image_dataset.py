from __future__ import annotations

from typing import Any

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from multimodal_ai.config.settings import settings


class ImageDataset(Dataset):
    """Loads images from a DataFrame path column and applies an optional transform,
    substituting a white RGB placeholder of size `settings.IMAGE_SIZE` when a
    file cannot be opened.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        path_column: str,
        transform: Any | None = None,
    ) -> None:
        """
        Build an image dataset from file paths stored in a DataFrame column.

        Args:
            df: DataFrame containing image file paths.
            path_column: Column name in `df` that holds the file paths.
            transform: Optional callable applied to each loaded image.
        """

        self.df = df.reset_index(drop=True)
        self.image_paths: list[str] = df[path_column].astype(str).tolist()
        self.transform = transform
        self.image_size = settings.IMAGE_SIZE

    def __len__(self) -> int:
        """Return the number of images in the dataset.

        Returns:
            int: Number of images.
        """

        return len(self.image_paths)

    def __getitem__(self, index: int) -> Any:
        """Load and return the transformed image at the given index.

        Args:
            index: Position of the image to load.

        Returns:
            Any: Transformed image, or a white RGB image of size
            `settings.IMAGE_SIZE` if loading fails.
        """

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

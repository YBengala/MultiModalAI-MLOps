from __future__ import annotations

import io
from typing import Callable, cast

import numpy as np
import torch
from PIL import Image

from multimodal_ai.config.settings import settings
from multimodal_ai.features.base_image_embedder import BaseImageEmbedder


class ImageEncoderInfer(BaseImageEmbedder):
    """Inference encoder for raw image bytes.

    Designed to process single images received via API requests.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        """Initializes the encoder.

        Args:
            model_name: Model identifier (default: settings.IMAGE_MODEL_NAME).
            device: 'cpu' or 'cuda' (default: settings.DEFAULT_DEVICE).
        """

        super().__init__(model_name=model_name, device=device)

    def _load_image(self, image_bytes: bytes) -> Image.Image:
        """Decodes bytes to RGB image. Returns white placeholder on failure.

        Args:
            image_bytes: Raw image data.

        Returns:
            Image.Image: Decoded image or white square fallback.
        """

        try:
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            size = settings.IMAGE_SIZE
            return Image.new("RGB", (size, size), (255, 255, 255))

    def encode_image_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Generates a 1D embedding from raw image bytes.

        Args:
            image_bytes: Raw image data to encode.

        Returns:
            np.ndarray: 1D embedding vector.
        """

        img = self._load_image(image_bytes)
        transform_fn = cast(Callable[[Image.Image], torch.Tensor], self.transform)
        x = transform_fn(img)
        x_batch = x.unsqueeze(0)
        batch_embedding = self.encode_tensor_batch(x_batch)

        return batch_embedding[0]

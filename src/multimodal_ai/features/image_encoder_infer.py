"""
Inference Image Encoder :
    - Specialized subclass for API usage (Single Image Processing).
    - Input : Raw bytes (from HTTP request).
    - Output : 1D Numpy array (Embedding vector).
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from multimodal_ai.config.settings import settings
from multimodal_ai.features.base_image_embedder import BaseImageEmbedder


class ImageEncoderInfer(BaseImageEmbedder):
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(model_name=model_name, device=device)

    def _load_image(self, image_bytes: bytes) -> Image.Image:
        try:
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            size = settings.IMAGE_SIZE
            return Image.new("RGB", (size, size), (255, 255, 255))

    def encode_image_bytes(self, image_bytes: bytes) -> np.ndarray:
        # Decodes bytes -> Transforms -> Encodes -> Returns Flat Vector
        img = self._load_image(image_bytes)
        x = self.transform(img).unsqueeze(0)

        return self.encode_tensor_batch(x)[0]

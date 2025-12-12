from __future__ import annotations

import numpy as np
import torch
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models import create_model
from torch.nn.functional import normalize

from multimodal_ai.config.settings import settings


class BaseImageEmbedder:
    """Wrapper for timm vision models with automatic preprocessing.

    This class handles the lifecycle of the vision backbone: loading weights,
    configuring the device (CPU/GPU), setting up the transformation pipeline,
    and managing batch inference.

    It serves as the parent class for both `ImageEncoderTrain` and `ImageEncoderInfer`.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        normalize_embeddings: bool | None = None,
    ) -> None:
        # Initialize configuration
        self.device = device or settings.DEFAULT_DEVICE
        self.batch_size = batch_size or settings.get_batch_size(self.device)
        self.normalize_embeddings = (
            normalize_embeddings
            if normalize_embeddings is not None
            else settings.IMAGE_NORMALIZE
        )

        # Load Vision Transformer model
        name = model_name or settings.IMAGE_MODEL_NAME
        self.model_name = name
        self.model = create_model(name, pretrained=True, num_classes=0)
        self.model.to(self.device)
        self.model.eval()

        # Transformation
        self.model_cfg = resolve_data_config(
            self.model.pretrained_cfg,
            model=self.model,
        )

        self.transform = create_transform(**self.model_cfg, is_training=False)

    @torch.no_grad()
    def encode_tensor_batch(
        self,
        image_batch: torch.Tensor,
    ) -> np.ndarray:
        """Encodes preprocessed tensors (B, C, H, W) into (B, D) numpy embeddings.

        Args:
            image_batch: Transformed tensor. Automatically moved to device.

        Returns:
            Embeddings array. L2-normalized if configured in __init__.
        """

        embeddings = self.model(image_batch.to(self.device))

        if self.normalize_embeddings:
            embeddings = normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

    def get_embedding_dim(self) -> int:
        """Returns the output feature dimension of the loaded backbone."""

        return getattr(self.model, "num_features", 0)

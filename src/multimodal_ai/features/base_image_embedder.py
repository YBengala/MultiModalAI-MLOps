"""
Image Feature Extractor :
    - Base class used for both `ImageEncoderTrain` and `ImageEncoderInfer`.
    - Input : (B, C, H, W) tensor.
    - Output : (B, D) numpy embeddings L2-normalized.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models import create_model
from torch.nn.functional import normalize

from multimodal_ai.config.settings import settings


class BaseImageEmbedder:
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        normalize_embeddings: bool | None = None,
    ) -> None:
        self.device = device or settings.DEFAULT_DEVICE
        self.batch_size = batch_size or settings.get_batch_size(self.device)
        self.normalize_embeddings = (
            normalize_embeddings
            if normalize_embeddings is not None
            else settings.IMAGE_NORMALIZE
        )
        name = model_name or settings.IMAGE_MODEL_NAME
        self.model_name = name
        self.model = create_model(name, pretrained=True, num_classes=0)
        self.model.to(self.device)
        self.model.eval()

        self.model_cfg = resolve_data_config(
            self.model.pretrained_cfg,
            model=self.model,
        )
        transform = create_transform(**self.model_cfg, is_training=False)
        if isinstance(transform, tuple):
            transform = transform[0]
        self.transform: Callable[..., Any] = transform

    @torch.no_grad()
    @torch.inference_mode()
    def encode_tensor_batch(
        self,
        image_batch: torch.Tensor,
    ) -> np.ndarray:
        embeddings = self.model(image_batch.to(self.device, non_blocking=True))
        if self.normalize_embeddings:
            embeddings = normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def get_embedding_dim(self) -> int:
        return getattr(self.model, "num_features", 0)

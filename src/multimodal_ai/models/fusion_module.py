"""
Model definition for multimodal fusion module :
    - FusionEmbeddings : concatenate image and text embeddings.
    - MultimodalMLP : multi-layer perceptron for multimodal fusion.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class FusionEmbeddings:
    def fuse_embeddings(self, img_emb: np.ndarray, txt_emb: np.ndarray) -> np.ndarray:
        # Dimensions check img vs text
        if img_emb.shape[0] != txt_emb.shape[0]:
            raise ValueError(
                f"Fusion Error: {img_emb.shape[0]} images vs {txt_emb.shape[0]} textes."
            )
        # concatenate img and text embeddings
        fusion_embeddings = np.concatenate((img_emb, txt_emb), axis=1)

        return fusion_embeddings.astype(np.float32)


class MultimodalMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout_rate: float = 0.3,
        hidden_l1: int = 1024,
        hidden_l2: int = 512,
        fn_activation: nn.Module = nn.ReLU(),
    ) -> None:
        super().__init__()

        self.classifier = nn.Sequential(
            # bloc 1
            nn.Linear(input_dim, hidden_l1),
            nn.BatchNorm1d(hidden_l1),
            fn_activation,
            nn.Dropout(p=dropout_rate),
            # bloc 2
            nn.Linear(hidden_l1, hidden_l2),
            nn.BatchNorm1d(hidden_l2),
            fn_activation,
            nn.Dropout(p=dropout_rate),
            # output
            nn.Linear(hidden_l2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

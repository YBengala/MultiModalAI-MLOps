from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class FusionEmbeddings:
    """class to concatenate pre-computed image and text embeddings."""

    def fuse_embeddings(self, img_emb: np.ndarray, txt_emb: np.ndarray) -> np.ndarray:
        """Merges image and text vectors via horizontal concatenation.

        Args:
            img_emb (np.ndarray): Image embeddings array of shape (N_samples, Img_Dim).
            txt_emb (np.ndarray): Text embeddings array of shape (N_samples, Txt_Dim).

        Returns:
            np.ndarray: The fused matrix of shape (N_samples, Img_Dim + Txt_Dim)
            in float32 format, ready for PyTorch conversion.

        Raises:
            ValueError: If the number of samples (rows) differs between image
                and text arrays.
        """
        # Dimensions check
        if img_emb.shape[0] != txt_emb.shape[0]:
            raise ValueError(
                f"Fusion Error: {img_emb.shape[0]} images vs {txt_emb.shape[0]} textes."
            )

        # concatenate img and text embeddings
        fusion_embeddings = np.concatenate((img_emb, txt_emb), axis=1)

        return fusion_embeddings.astype(np.float32)


class MultimodalMLP(nn.Module):
    """Deep Neural Network for multimodal classification (Rakuten).

    Implements a funnel-shaped architecture to progressively reduce dimensionality
    and extract high-level features from the fused input.

    Architecture:
        Input -> [Linear(2048) - BN - ReLU - Dropout]
              -> [Linear(1024) - BN - ReLU - Dropout]
              -> Output(Num_Classes)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        self.classifier = nn.Sequential(
            # bloc 1
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            # bloc 2
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            # output
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, Input_Dim).

        Returns:
            torch.Tensor: Raw logits of shape (Batch_Size, Num_Classes).
        """
        return self.classifier(x)

"""Learned embedding for aggregated object categories and normalized statistics."""

from __future__ import annotations

import torch
from torch import nn


class ObjectFeatureEncoder(nn.Module):
    """Combine class embeddings with count/confidence/box/persistence statistics."""

    def __init__(
        self,
        num_classes: int = 80,
        class_embedding_dim: int = 32,
        stats_dim: int = 8,
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        if min(num_classes, class_embedding_dim, stats_dim, output_dim) < 1:
            raise ValueError("all ObjectFeatureEncoder dimensions must be positive")
        self.num_classes = num_classes
        self.stats_dim = stats_dim
        self.class_embedding = nn.Embedding(num_classes, class_embedding_dim)
        self.stats_projection = nn.Sequential(
            nn.Linear(stats_dim, class_embedding_dim), nn.ReLU(), nn.Linear(class_embedding_dim, class_embedding_dim)
        )
        self.output_projection = nn.Sequential(
            nn.Linear(class_embedding_dim, output_dim), nn.LayerNorm(output_dim), nn.GELU()
        )

    def forward(
        self,
        class_ids: torch.Tensor,
        stats: torch.Tensor,
        object_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode padded class/stat arrays into one video-level vector per batch item."""
        if class_ids.ndim != 2 or stats.ndim != 3:
            raise ValueError("expected class_ids [B,N] and stats [B,N,S]")
        if class_ids.shape != stats.shape[:2] or stats.shape[-1] != self.stats_dim:
            raise ValueError("class_ids and stats shapes do not match encoder configuration")
        if class_ids.numel() and (class_ids.min() < 0 or class_ids.max() >= self.num_classes):
            raise ValueError(f"class IDs must be in [0, {self.num_classes - 1}]")
        encoded = self.class_embedding(class_ids) + self.stats_projection(stats.float())
        if object_mask is None:
            object_mask = torch.ones_like(class_ids, dtype=torch.bool)
        if object_mask.shape != class_ids.shape:
            raise ValueError("object_mask must have shape [B,N]")
        mask = object_mask.to(dtype=encoded.dtype).unsqueeze(-1)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        has_objects = mask.sum(dim=1) > 0
        pooled = torch.where(has_objects, pooled, torch.zeros_like(pooled))
        return self.output_projection(pooled) * has_objects.to(pooled.dtype)

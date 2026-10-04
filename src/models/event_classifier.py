"""Optional binary normal/abnormal classifier operating on temporal embeddings."""

from __future__ import annotations

import torch
from torch import nn


class EventClassifier(nn.Module):
    """Separate auxiliary event head; its labels are never used as free-form captions."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or not 0.0 <= dropout < 1.0:
            raise ValueError("input_dim/hidden_dim must be positive and dropout must be in [0,1)")
        self.event_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, temporal_features: torch.Tensor) -> dict[str, torch.Tensor]:
        if temporal_features.ndim != 2:
            raise ValueError("temporal_features must be shaped [batch, feature_dim]")
        event_embedding = self.event_embedding(temporal_features)
        logits = self.classifier(event_embedding)
        return {
            "event_embedding": event_embedding,
            "logits": logits,
            "probabilities": torch.softmax(logits, dim=-1),
        }

    @staticmethod
    def predict(probabilities: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return 0=normal or 1=abnormal using a configurable abnormal threshold."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        return (probabilities[..., 1] >= threshold).long()

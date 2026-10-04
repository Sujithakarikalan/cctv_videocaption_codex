"""SlowFast-R50 temporal encoder with a clearly reported frame-aggregation fallback."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

LOGGER = logging.getLogger(__name__)


class FrameAggregationEncoder(nn.Module):
    """Small trainable 2D CNN that mean/max aggregates per-frame features over time."""

    def __init__(self, output_dim: int = 512) -> None:
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, output_dim // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_dim // 2), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.output_dim = output_dim

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = video.shape
        per_frame = self.frame_encoder(video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width))
        per_frame = per_frame.flatten(1).view(batch, frames, -1)
        return torch.cat((per_frame.mean(dim=1), per_frame.max(dim=1).values), dim=1)


class SlowFastEncoder(nn.Module):
    """Expose pre-classifier Kinetics features from PyTorchVideo SlowFast-R50.

    ``allow_fallback`` only applies when the PyTorchVideo dependency is unavailable. If
    used, the fallback backend is printed/logged explicitly and is not presented as SlowFast.
    """

    def __init__(
        self,
        pretrained: bool = True,
        allow_fallback: bool = True,
        alpha: int = 4,
        beta_inv: int = 8,
        fallback_dim: int = 512,
        model_factory: Any | None = None,
    ) -> None:
        super().__init__()
        if alpha < 1 or beta_inv < 1:
            raise ValueError("alpha and beta_inv must be positive")
        self.alpha = alpha
        self.beta_inv = beta_inv
        self.pretrained = pretrained
        self.model: nn.Module | None = None
        self._feature_buffer: list[torch.Tensor] = []
        self._hook_handle: Any | None = None

        if model_factory is None:
            try:
                from pytorchvideo.models.hub import slowfast_r50
            except (ImportError, OSError) as exc:
                if not allow_fallback:
                    raise RuntimeError(
                        "SlowFast-R50 requires PyTorchVideo. Install a compatible pytorchvideo build "
                        "or set allow_fallback: true to use the explicitly reported frame aggregator."
                    ) from exc
                LOGGER.warning(
                    "PyTorchVideo SlowFast-R50 is unavailable (%s). Using FrameAggregationEncoder fallback; "
                    "this is not a SlowFast model and has no Kinetics pretrained weights.", exc,
                )
                print(
                    "WARNING: PyTorchVideo SlowFast-R50 unavailable; using explicit frame-aggregation "
                    "fallback (not pretrained SlowFast)."
                )
                self.fallback = FrameAggregationEncoder(fallback_dim)
                self.backend = "frame_aggregation_fallback"
                self.embedding_dim = fallback_dim
                self._install_normalization()
                return
            model_factory = slowfast_r50

        self.model = model_factory(pretrained=pretrained)
        self.backend = "pytorchvideo_slowfast_r50"
        self.fallback = None
        head = self.model.blocks[-1] if hasattr(self.model, "blocks") else None
        projection = getattr(head, "proj", None)
        if projection is None:
            raise RuntimeError("Unexpected PyTorchVideo SlowFast-R50 architecture: final head has no proj layer")
        self._hook_handle = projection.register_forward_pre_hook(self._capture_pre_classifier)
        self.embedding_dim = int(getattr(projection, "in_channels", getattr(projection, "in_features", 0)))
        if self.embedding_dim < 1:
            raise RuntimeError("Could not infer SlowFast pre-classifier feature dimension from final projection")
        self._install_normalization()

    def _install_normalization(self) -> None:
        self.register_buffer("mean", torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1, 1), persistent=False)

    def _capture_pre_classifier(self, _module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        self._feature_buffer.append(inputs[0])

    def build_pathways(self, video: torch.Tensor) -> list[torch.Tensor]:
        """Convert normalized ``[B,C,T,H,W]`` input into SlowFast slow/fast pathways."""
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"Expected video shaped [B,3,T,H,W], got {tuple(video.shape)}")
        normalized = (video.float() - self.mean) / self.std
        slow = normalized[:, :, :: self.alpha]
        return [slow.contiguous(), normalized.contiguous()]

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Return one temporal embedding per video clip, excluding Kinetics logits."""
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"Expected video shaped [B,3,T,H,W], got {tuple(video.shape)}")
        x = video.float()
        if x.numel() and x.max().item() > 1.0:
            x = x / 255.0
        if self.fallback is not None:
            normalized = (x - self.mean) / self.std
            return self.fallback(normalized)
        self._feature_buffer.clear()
        self.model(self.build_pathways(x))
        if not self._feature_buffer:
            raise RuntimeError("SlowFast forward did not capture the final pre-classifier feature tensor")
        features = self._feature_buffer[-1].flatten(1)
        if features.shape[1] != self.embedding_dim:
            raise RuntimeError(
                f"SlowFast feature dimension mismatch: expected {self.embedding_dim}, got {features.shape[1]}"
            )
        self._feature_buffer.clear()
        return features

    def close(self) -> None:
        """Remove the classifier hook when an encoder is discarded."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

"""Torchvision ResNet18 frame encoder returning normalized 512-D visual features."""

from __future__ import annotations

import logging

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

LOGGER = logging.getLogger(__name__)


class ResNet18Encoder(nn.Module):
    """ResNet18 without its classification layer, with ImageNet preprocessing."""

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True) -> None:
        super().__init__()
        self.pretrained = pretrained
        self.weights_name = "ResNet18_Weights.DEFAULT" if pretrained else "untrained-random-initialization"
        self.weights_url = ResNet18_Weights.DEFAULT.url if pretrained else None
        try:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            model = resnet18(weights=weights)
        except Exception as exc:
            raise RuntimeError(
                "Could not initialize pretrained ResNet18 weights. Check network/cache, or use "
                "--no-pretrained only for a smoke test."
            ) from exc
        model.fc = nn.Identity()
        self.model = model
        self.embedding_dim = 512
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for parameter in self.parameters():
                parameter.requires_grad = False
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode RGB images shaped ``[B,3,H,W]`` and L2-normalize their 512-D features."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected images shaped [batch, 3, height, width], got {tuple(images.shape)}")
        x = images.float()
        if x.numel() and x.max().item() > 1.0:
            x = x / 255.0
        x = (x - self.mean) / self.std
        features = self.model(x)
        return torch.nn.functional.normalize(features, p=2, dim=1)


def encode_bgr_frames(
    encoder: ResNet18Encoder,
    frames_bgr: list[object],
    device: torch.device,
    frame_size: int = 224,
    batch_size: int = 16,
) -> torch.Tensor:
    """Resize OpenCV BGR frames to the configured square input and encode in batches."""
    import cv2
    import numpy as np

    if frame_size < 1 or batch_size < 1:
        raise ValueError("frame_size and batch_size must be positive")
    if not frames_bgr:
        return torch.empty((0, encoder.embedding_dim), dtype=torch.float32)
    encoder = encoder.to(device)
    encoder.eval()
    all_features: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(frames_bgr), batch_size):
            batch = []
            for bgr in frames_bgr[start : start + batch_size]:
                rgb = cv2.cvtColor(np.asarray(bgr), cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (frame_size, frame_size), interpolation=cv2.INTER_AREA)
                batch.append(torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1))
            tensor = torch.stack(batch).to(device=device, dtype=torch.float32)
            all_features.append(encoder(tensor).cpu())
    return torch.cat(all_features, dim=0)

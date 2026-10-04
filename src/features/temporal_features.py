"""Utilities for turning one or more sampled clips into video-level embeddings."""

from __future__ import annotations

import torch

from src.models.slowfast_encoder import SlowFastEncoder


def encode_video_clips(
    encoder: SlowFastEncoder,
    clips: torch.Tensor,
    device: torch.device,
    batch_size: int = 4,
) -> torch.Tensor:
    """Encode `[K,C,T,H,W]` or `[B,K,C,T,H,W]` clips and average per source video."""
    if clips.ndim == 5:
        clips = clips.unsqueeze(0)
    if clips.ndim != 6 or batch_size < 1:
        raise ValueError("clips must have shape [K,C,T,H,W] or [B,K,C,T,H,W], batch_size > 0")
    batch, clip_count, channels, frames, height, width = clips.shape
    flattened = clips.reshape(batch * clip_count, channels, frames, height, width)
    encoded: list[torch.Tensor] = []
    encoder = encoder.to(device)
    encoder.eval()
    with torch.inference_mode():
        for start in range(0, flattened.shape[0], batch_size):
            encoded.append(encoder(flattened[start : start + batch_size].to(device)).cpu())
    features = torch.cat(encoded, dim=0).view(batch, clip_count, -1).mean(dim=1)
    return features

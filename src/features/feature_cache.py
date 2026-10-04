"""Versioned per-video PyTorch feature cache with source and config validation."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import torch

LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = 2


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file incrementally to avoid loading long videos into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class FeatureCache:
    """Read/write video feature tensors, rejecting stale model/source configurations."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, video_id: str) -> Path:
        safe_id = hashlib.sha256(video_id.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{safe_id}.pt"

    def load(
        self,
        video_id: str,
        source_hash: str,
        parameters: dict[str, Any],
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any] | None:
        path = self.path_for(video_id)
        if not path.is_file():
            return None
        try:
            payload = torch.load(path, map_location=map_location, weights_only=True)
        except Exception as exc:
            LOGGER.warning("Ignoring unreadable feature cache %s: %s", path, exc)
            return None
        if payload.get("schema_version") != SCHEMA_VERSION:
            LOGGER.info("Ignoring feature cache with incompatible schema: %s", path)
            return None
        metadata = payload.get("metadata", {})
        if metadata.get("encoder_name") != "resnet18":
            LOGGER.info("Ignoring non-ResNet18 feature cache for video %s", video_id)
            return None
        if metadata.get("source_sha256") != source_hash or metadata.get("parameters") != parameters:
            LOGGER.info("Ignoring stale feature cache for video %s", video_id)
            return None
        return payload

    def save(
        self,
        video_id: str,
        source_hash: str,
        parameters: dict[str, Any],
        tensors: dict[str, torch.Tensor | list[str]],
        extra_metadata: dict[str, Any] | None = None,
    ) -> Path:
        path = self.path_for(video_id)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "metadata": {
                "video_id": video_id,
                "source_video_path": str(extra_metadata.get("source_video_path", "")) if extra_metadata else "",
                "source_sha256": source_hash,
                "parameters": parameters,
                "encoder_name": "resnet18",
                "feature_dimension": 512,
                "pretrained": bool(parameters.get("pretrained", False)),
                "preprocessing": parameters.get("preprocessing", {}),
                "weights": parameters.get("weights", "unknown"),
                "semantic_similarity_threshold": parameters.get("semantic_similarity_threshold"),
                "motion_threshold": parameters.get("motion_threshold"),
                "cache_schema_version": SCHEMA_VERSION,
                "feature_shapes": {
                    key: list(value.shape) for key, value in tensors.items() if isinstance(value, torch.Tensor)
                },
                **(extra_metadata or {}),
            },
            "features": {
                key: value.detach().cpu().contiguous() if isinstance(value, torch.Tensor) else value
                for key, value in tensors.items()
            },
        }
        temporary = path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        return path


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write readable UTF-8 JSON atomically."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(destination)

"""Pretrained SlowFast-R50 Kinetics-400 clip classification."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from torch import nn

LABELS_URL = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"


@dataclass(frozen=True)
class ActionPrediction:
    class_id: int
    label: str
    confidence: float


def preprocess_frames(frames_bgr: Sequence[np.ndarray], size: int = 256) -> torch.Tensor:
    """Resize/crop and normalize a list of BGR frames to [1,3,T,H,W]."""
    if not frames_bgr:
        raise ValueError("At least one video frame is required")
    processed: list[np.ndarray] = []
    for frame in frames_bgr:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Frames must be valid BGR images shaped [height,width,3]")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        scale = size / min(height, width)
        new_w, new_h = int(round(width * scale)), int(round(height * scale))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        left, top = (new_w - size) // 2, (new_h - size) // 2
        processed.append(resized[top:top + size, left:left + size])
    array = np.stack(processed).astype(np.float32) / 255.0
    video = torch.from_numpy(array).permute(3, 0, 1, 2).unsqueeze(0)
    mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1, 1)
    return (video - mean) / std


def pack_slowfast_pathways(video: torch.Tensor, alpha: int = 4) -> list[torch.Tensor]:
    """Pack normalized [B,C,T,H,W] input as SlowFast pathway inputs."""
    if video.ndim != 5 or video.shape[1] != 3 or alpha < 1:
        raise ValueError("Expected [B,3,T,H,W] input and positive alpha")
    count = max(1, video.shape[2] // alpha)
    indices = torch.linspace(0, video.shape[2] - 1, count, device=video.device).long()
    return [torch.index_select(video, 2, indices).contiguous(), video.contiguous()]


class SlowFastActionRecognizer:
    """Load official PyTorchVideo SlowFast-R50 Kinetics-400 weights and return top-k."""

    def __init__(
        self,
        device: str = "auto",
        model: nn.Module | None = None,
        labels: Sequence[str] | None = None,
        project_root: str | Path | None = None,
    ) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[2])
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but this PyTorch build/device has no CUDA support")
        self.device = torch.device(device)
        cache_dir = self.project_root / "checkpoints" / "torch_hub"
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(cache_dir))
        self.labels = list(labels) if labels is not None else self._load_labels(cache_dir)
        if model is None:
            try:
                from pytorchvideo.models.hub import slowfast_r50
            except (ImportError, OSError) as exc:
                raise RuntimeError("Install PyTorchVideo with `python -m pip install pytorchvideo` first") from exc
            # This is the official pretrained Kinetics-400 model, not the Phase 4 fallback encoder.
            model = slowfast_r50(pretrained=True, progress=True)
        self.model = model.eval().to(self.device)
        self.backend = "pytorchvideo_slowfast_r50_kinetics400"

    def _load_labels(self, cache_dir: Path) -> list[str]:
        path = cache_dir / "kinetics_classnames.json"
        if not path.exists():
            try:
                urllib.request.urlretrieve(LABELS_URL, path)
            except Exception as exc:
                raise RuntimeError(f"Could not download official Kinetics-400 labels from {LABELS_URL}: {exc}") from exc
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return [str(item) for item in raw]
        labels = [""] * len(raw)
        for name, index in raw.items():
            labels[int(index)] = str(name).replace('"', "")
        if any(not label for label in labels):
            raise ValueError("Kinetics label map contains missing class ids")
        return labels

    @torch.inference_mode()
    def predict_frames(self, frames_bgr: Sequence[np.ndarray], top_k: int = 5) -> list[ActionPrediction]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        clip = preprocess_frames(frames_bgr).to(self.device)
        pathways = [pathway for pathway in pack_slowfast_pathways(clip)]
        logits = self.model([pathway for pathway in pathways])
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise RuntimeError(f"Expected model logits [1,num_classes], got {tuple(logits.shape)}")
        if logits.shape[1] != len(self.labels):
            raise RuntimeError(f"Model returned {logits.shape[1]} classes but label map has {len(self.labels)}")
        probabilities = torch.softmax(logits[0], dim=-1)
        values, indices = probabilities.topk(min(top_k, len(self.labels)))
        return [
            ActionPrediction(int(index), self.labels[int(index)], float(score))
            for score, index in zip(values.cpu(), indices.cpu())
        ]

    def predict_tensor(self, video: torch.Tensor, top_k: int = 5) -> list[ActionPrediction]:
        """Predict from pre-normalized [1,3,T,H,W] RGB video tensor (test/integration API)."""
        if video.ndim != 5 or video.shape[:2] != (1, 3):
            raise ValueError("video must have shape [1,3,T,H,W]")
        logits = self.model(pack_slowfast_pathways(video.to(self.device)))
        if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != len(self.labels):
            raise RuntimeError("SlowFast logits do not align with the configured Kinetics labels")
        probs = logits.softmax(-1)[0]
        values, indices = probs.topk(min(top_k, len(self.labels)))
        return [ActionPrediction(int(i), self.labels[int(i)], float(v)) for v, i in zip(values.cpu(), indices.cpu())]


def predictions_to_dict(predictions: Sequence[ActionPrediction]) -> list[dict[str, Any]]:
    return [asdict(item) for item in predictions]

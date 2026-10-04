"""Phase 4 temporal data/model/event tests; no pretrained model weights are downloaded."""

import csv
import builtins
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from src.data.temporal_dataset import TemporalRecord, TemporalVideoDataset, load_temporal_csv
from src.evaluation.classification_metrics import binary_classification_metrics, select_event_threshold
from src.features.temporal_features import encode_video_clips
from src.models.event_classifier import EventClassifier
from src.models.slowfast_encoder import SlowFastEncoder
from src.training.train_event_classifier import train_event_classifier


def make_video(path: Path, seed: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (48, 32))
    if not writer.isOpened():
        pytest.skip("This OpenCV build has no MP4V encoder")
    rng = np.random.default_rng(seed)
    for frame_index in range(8):
        frame = np.full((32, 48, 3), seed * 30, dtype=np.uint8)
        cv2.circle(frame, (frame_index * 5 + 4, 16), 5, tuple(int(x) for x in rng.integers(20, 240, 3)), -1)
        writer.write(frame)
    writer.release()
    return path


def test_temporal_csv_split_leakage_is_rejected(tmp_path: Path) -> None:
    video_a = make_video(tmp_path / "a.mp4", 0)
    video_b = make_video(tmp_path / "b.mp4", 1)
    manifest = tmp_path / "events.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["video_id", "video_path", "label", "split"])
        writer.writeheader()
        writer.writerows([
            {"video_id": "same-source", "video_path": video_a.name, "label": "normal", "split": "train"},
            {"video_id": "same-source", "video_path": video_b.name, "label": "abnormal", "split": "test"},
        ])
    with pytest.raises(ValueError, match="leakage"):
        load_temporal_csv(manifest)


def test_temporal_dataset_returns_uniform_video_clips(tmp_path: Path) -> None:
    video = make_video(tmp_path / "short.mp4", 1)
    record = TemporalRecord("short", video, "normal", "train")
    dataset = TemporalVideoDataset([record], "train", num_frames=4, sampling_rate=2, frame_size=32, clips_per_video=2)
    item = dataset[0]
    assert item["clips"].shape == (2, 3, 4, 32, 32)
    assert item["clips"].dtype == torch.uint8
    assert item["label"] == 0


def test_slowfast_pathways_and_preclassifier_embedding_with_injected_model() -> None:
    class FakeHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Conv3d(3, 5, kernel_size=1)

    class FakeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([nn.Identity(), FakeHead()])

        def forward(self, pathways):
            slow, fast = pathways
            pooled = (slow.mean(dim=2, keepdim=True) + fast.mean(dim=2, keepdim=True)) / 2
            pooled = pooled.mean(dim=(-1, -2), keepdim=True)
            return self.blocks[-1].proj(pooled)

    encoder = SlowFastEncoder(pretrained=False, alpha=4, model_factory=lambda pretrained: FakeModel())
    clip = torch.randint(0, 255, (2, 3, 8, 32, 32), dtype=torch.uint8)
    slow, fast = encoder.build_pathways(clip)
    embedding = encoder(clip)
    assert slow.shape[2] == 2 and fast.shape[2] == 8
    assert embedding.shape == (2, 3)


def test_missing_pytorchvideo_uses_explicit_fallback(capsys) -> None:
    encoder = SlowFastEncoder(pretrained=False, allow_fallback=True, fallback_dim=64)
    if encoder.backend != "frame_aggregation_fallback":
        pytest.skip("PyTorchVideo is installed; fallback is not active in this environment")
    assert "not pretrained SlowFast" in capsys.readouterr().out
    features = encoder(torch.randint(0, 255, (1, 3, 4, 32, 32), dtype=torch.uint8))
    assert features.shape == (1, 64)


def test_event_classifier_metrics_threshold_and_embedding() -> None:
    classifier = EventClassifier(input_dim=6, hidden_dim=8)
    output = classifier(torch.randn(4, 6))
    assert output["logits"].shape == (4, 2)
    assert output["event_embedding"].shape == (4, 8)
    labels = [0, 0, 1, 1]
    probabilities = [0.1, 0.2, 0.8, 0.9]
    threshold = select_event_threshold(labels, probabilities)
    metrics = binary_classification_metrics(labels, probabilities, threshold)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def _force_optional_pytorchvideo_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def import_without_pytorchvideo(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pytorchvideo.models.hub":
            raise ImportError("test intentionally exercises the documented fallback")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_pytorchvideo)


def test_temporal_features_mean_pool_clips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _force_optional_pytorchvideo_missing(monkeypatch)
    encoder = SlowFastEncoder(pretrained=False, allow_fallback=True, fallback_dim=32)
    clips = torch.randint(0, 255, (2, 3, 4, 32, 32), dtype=torch.uint8)
    features = encode_video_clips(encoder, clips, torch.device("cpu"), batch_size=1)
    assert features.shape == (1, 32)


def test_event_training_one_epoch_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _force_optional_pytorchvideo_missing(monkeypatch)
    rows = []
    for split, seed, label in [
        ("train", 1, "normal"), ("train", 2, "abnormal"),
        ("val", 3, "normal"), ("val", 4, "abnormal"),
    ]:
        path = make_video(tmp_path / f"{split}_{seed}.mp4", seed)
        rows.append({"video_id": path.stem, "video_path": path.name, "label": label, "split": split})
    manifest = tmp_path / "events.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["video_id", "video_path", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)
    fallback = SlowFastEncoder(pretrained=False, allow_fallback=True, fallback_dim=32)
    result = train_event_classifier(
        manifest_path=manifest, output_dir=tmp_path / "ckpt", epochs=1, batch_size=2,
        num_frames=4, sampling_rate=1, frame_size=32, clips_per_video=1, alpha=2,
        pretrained_temporal=False, fine_tune_temporal=True, device_name="cpu", workers=0,
        temporal_encoder=fallback,
    )
    assert Path(result["best_checkpoint"]).is_file()
    assert Path(result["last_checkpoint"]).is_file()
    assert result["temporal_backend"] == fallback.backend

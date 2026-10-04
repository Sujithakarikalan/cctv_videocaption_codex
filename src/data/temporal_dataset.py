"""Video-level temporal dataset adapters for CSV manifests and class directories."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass(frozen=True)
class TemporalRecord:
    video_id: str
    video_path: Path
    label: str
    split: str


def load_temporal_csv(manifest_path: str | Path) -> list[TemporalRecord]:
    """Load rows ``video_path,label,split`` and reject cross-split video leakage."""
    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Temporal manifest does not exist: {manifest}")
    records: list[TemporalRecord] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"video_path", "label", "split"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Temporal CSV must contain columns: {', '.join(sorted(required))}")
        for row_number, row in enumerate(reader, start=2):
            raw_path = Path(row["video_path"].strip()).expanduser()
            video_path = (manifest.parent / raw_path).resolve() if not raw_path.is_absolute() else raw_path.resolve()
            label, split = row["label"].strip(), row["split"].strip().lower()
            if not video_path.is_file():
                raise FileNotFoundError(f"Manifest row {row_number} video does not exist: {video_path}")
            if not label or split not in {"train", "val", "test"}:
                raise ValueError(f"Manifest row {row_number} requires a label and split in train/val/test")
            video_id = row.get("video_id", "").strip() or video_path.stem
            records.append(TemporalRecord(video_id, video_path, label, split))
    _check_video_split_leakage(records)
    return records


def load_event_directories(root: str | Path) -> list[TemporalRecord]:
    """Read ``root/{train,val,test}/{normal,abnormal}/video`` class directories."""
    dataset_root = Path(root).expanduser().resolve()
    records = []
    for split in ("train", "val", "test"):
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            for video in sorted(path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES):
                records.append(TemporalRecord(video.stem, video.resolve(), class_dir.name.lower(), split))
    if not records:
        raise ValueError(f"No video files found in event class directories under {dataset_root}")
    _check_video_split_leakage(records)
    return records


def _check_video_split_leakage(records: list[TemporalRecord]) -> None:
    splits_by_identity: dict[str, set[str]] = {}
    for record in records:
        identities = (f"path:{record.video_path.resolve()}".casefold(), f"video_id:{record.video_id}".casefold())
        for identity in identities:
            splits_by_identity.setdefault(identity, set()).add(record.split)
    leaked = [identity for identity, splits in splits_by_identity.items() if len(splits) > 1]
    if leaked:
        raise ValueError(f"Video identity leakage across splits detected for: {leaked[:5]}")


class TemporalVideoDataset(Dataset[dict[str, Any]]):
    """Uniformly sample short clips from each source video for SlowFast/event training."""

    def __init__(
        self,
        records: list[TemporalRecord],
        split: str,
        num_frames: int = 32,
        sampling_rate: int = 2,
        frame_size: int = 224,
        clips_per_video: int = 1,
        label_to_id: dict[str, int] | None = None,
    ) -> None:
        if min(num_frames, sampling_rate, frame_size, clips_per_video) < 1:
            raise ValueError("num_frames, sampling_rate, frame_size and clips_per_video must be positive")
        self.records = [record for record in records if record.split == split]
        if not self.records:
            raise ValueError(f"No temporal videos assigned to split {split!r}")
        labels = sorted({record.label for record in records})
        if {label.casefold() for label in labels} >= {"normal", "abnormal"}:
            default_mapping = {"normal": 0, "abnormal": 1}
        else:
            default_mapping = {label: index for index, label in enumerate(labels)}
        self.label_to_id = label_to_id or default_mapping
        missing_labels = {record.label for record in self.records} - self.label_to_id.keys()
        if missing_labels:
            raise ValueError(f"Label mapping is missing classes: {sorted(missing_labels)}")
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.frame_size = frame_size
        self.clips_per_video = clips_per_video

    def __len__(self) -> int:
        return len(self.records)

    def _read_frame(self, capture: cv2.VideoCapture, frame_index: int, path: Path) -> np.ndarray:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            # Fallback for codecs that reject direct seeking: reopen and decode sequentially.
            capture.release()
            capture.open(str(path))
            if not capture.isOpened():
                raise ValueError(f"OpenCV could not reopen video: {path}")
            for _ in range(frame_index + 1):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Could not decode frame {frame_index} from video: {path}")
        return frame

    def _sample_video(self, path: Path) -> torch.Tensor:
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError(f"OpenCV could not open video (unsupported codec or corrupt file): {path}")
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count < 1:
                raise ValueError(f"Video has no decodable frame metadata: {path}")
            clip_span = (self.num_frames - 1) * self.sampling_rate
            max_start = max(frame_count - 1 - clip_span, 0)
            starts = np.linspace(0, max_start, num=self.clips_per_video, dtype=np.int64)
            clips = []
            for start in starts:
                indices = [min(int(start + offset * self.sampling_rate), frame_count - 1) for offset in range(self.num_frames)]
                frames = []
                for index in indices:
                    frame = self._read_frame(capture, index, path)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(rgb, (self.frame_size, self.frame_size), interpolation=cv2.INTER_AREA)
                    frames.append(torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1))
                clips.append(torch.stack(frames, dim=1))  # C,T,H,W uint8
            return torch.stack(clips, dim=0)
        finally:
            capture.release()

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "video_id": record.video_id,
            "video_path": str(record.video_path),
            "split": record.split,
            "label": self.label_to_id[record.label],
            "clips": self._sample_video(record.video_path),
        }

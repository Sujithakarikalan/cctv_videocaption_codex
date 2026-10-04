"""Aggregate YOLO detections into object counts, confidence, boxes, and persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch

from src.models.yolo_extractor import ObjectDetection


@dataclass
class ObjectClassSummary:
    class_id: int
    label: str
    count: int
    confidence_mean: float
    confidence_max: float
    persistence: float
    bbox_xyxy_mean: tuple[float, float, float, float]
    track_ids: list[int]

    def stats(self, frame_count: int) -> list[float]:
        """Return eight normalized numeric values for ``ObjectFeatureEncoder``."""
        x1, y1, x2, y2 = self.bbox_xyxy_mean
        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        width, height = max(0.0, x2 - x1), max(0.0, y2 - y1)
        return [
            self.count / max(frame_count, 1), self.confidence_mean, self.confidence_max,
            center_x, center_y, width, height, self.persistence,
        ]


@dataclass
class ObjectFeatureSummary:
    frame_count: int
    counts_per_frame: list[int]
    total_detections: int
    mean_objects_per_frame: float
    max_objects_per_frame: int
    class_counts: dict[str, int]
    objects: list[ObjectClassSummary]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "counts_per_frame": self.counts_per_frame,
            "total_detections": self.total_detections,
            "mean_objects_per_frame": self.mean_objects_per_frame,
            "max_objects_per_frame": self.max_objects_per_frame,
            "class_counts": self.class_counts,
            "objects": [asdict(item) for item in self.objects],
        }

    def to_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``class_ids [1,N]``, ``stats [1,N,8]``, and valid-object mask [1,N]."""
        if not self.objects:
            return (
                torch.zeros((1, 1), dtype=torch.long),
                torch.zeros((1, 1, 8), dtype=torch.float32),
                torch.zeros((1, 1), dtype=torch.bool),
            )
        ids = torch.tensor([[obj.class_id for obj in self.objects]], dtype=torch.long)
        stats = torch.tensor([[obj.stats(self.frame_count) for obj in self.objects]], dtype=torch.float32)
        mask = torch.ones((1, len(self.objects)), dtype=torch.bool)
        return ids, stats, mask


def aggregate_object_features(
    per_frame_detections: Sequence[Sequence[ObjectDetection]],
) -> ObjectFeatureSummary:
    """Summarize detections by category and calculate per-video temporal persistence."""
    frame_count = len(per_frame_detections)
    counts = [len(detections) for detections in per_frame_detections]
    grouped: dict[int, dict[str, Any]] = {}
    for frame_index, detections in enumerate(per_frame_detections):
        frame_classes = {detection.class_id for detection in detections}
        for detection in detections:
            record = grouped.setdefault(detection.class_id, {
                "label": detection.label, "confidences": [], "boxes": [],
                "frames": set(), "track_ids": set(), "count": 0,
            })
            record["count"] += 1
            record["confidences"].append(float(detection.confidence))
            record["boxes"].append(detection.bbox_xyxy_normalized)
            record["frames"].add(frame_index)
            if detection.track_id is not None:
                record["track_ids"].add(int(detection.track_id))
        # `frame_classes` is kept explicit for clarity: persistence counts distinct frames.
        for class_id in frame_classes:
            grouped[class_id]["frames"].add(frame_index)

    objects: list[ObjectClassSummary] = []
    for class_id, record in sorted(grouped.items()):
        boxes = torch.tensor(record["boxes"], dtype=torch.float32)
        objects.append(ObjectClassSummary(
            class_id=class_id,
            label=record["label"],
            count=record["count"],
            confidence_mean=float(sum(record["confidences"]) / record["count"]),
            confidence_max=float(max(record["confidences"])),
            persistence=len(record["frames"]) / max(frame_count, 1),
            bbox_xyxy_mean=tuple(float(value) for value in boxes.mean(dim=0).tolist()),
            track_ids=sorted(record["track_ids"]),
        ))
    return ObjectFeatureSummary(
        frame_count=frame_count,
        counts_per_frame=counts,
        total_detections=sum(counts),
        mean_objects_per_frame=(sum(counts) / frame_count) if frame_count else 0.0,
        max_objects_per_frame=max(counts, default=0),
        class_counts={item.label: item.count for item in objects},
        objects=objects,
    )

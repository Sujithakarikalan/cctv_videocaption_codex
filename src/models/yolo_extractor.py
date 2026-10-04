"""Ultralytics YOLOv8 pretrained detection wrapper with optional object tracking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import os

from src.utils.device import get_device


@dataclass(frozen=True)
class ObjectDetection:
    class_id: int
    label: str
    confidence: float
    bbox_xyxy_normalized: tuple[float, float, float, float]
    track_id: int | None = None


def parse_ultralytics_result(result: Any, frame_width: int, frame_height: int) -> list[ObjectDetection]:
    """Convert one Ultralytics result into stable normalized detection records."""
    if frame_width < 1 or frame_height < 1:
        raise ValueError("frame dimensions must be positive")
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy() if hasattr(boxes.xyxy, "detach") else np.asarray(boxes.xyxy)
    classes = boxes.cls.detach().cpu().numpy() if hasattr(boxes.cls, "detach") else np.asarray(boxes.cls)
    confidences = boxes.conf.detach().cpu().numpy() if hasattr(boxes.conf, "detach") else np.asarray(boxes.conf)
    tracks = getattr(boxes, "id", None)
    if tracks is not None:
        tracks = tracks.detach().cpu().numpy() if hasattr(tracks, "detach") else np.asarray(tracks)
    names = getattr(result, "names", {})
    detections: list[ObjectDetection] = []
    for index, (box, class_value, confidence) in enumerate(zip(xyxy, classes, confidences)):
        class_id = int(class_value)
        x1, y1, x2, y2 = (float(box[0]) / frame_width, float(box[1]) / frame_height,
                          float(box[2]) / frame_width, float(box[3]) / frame_height)
        normalized = tuple(float(np.clip(value, 0.0, 1.0)) for value in (x1, y1, x2, y2))
        label = str(names.get(class_id, class_id) if isinstance(names, dict) else names[class_id])
        track_id = int(tracks[index]) if tracks is not None and index < len(tracks) else None
        detections.append(ObjectDetection(class_id, label, float(confidence), normalized, track_id))
    return detections


class YOLODetectionExtractor:
    """Run pretrained YOLOv8 detection on a frame sequence; weights load lazily."""

    def __init__(
        self,
        weights: str | Path = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        device: str = "auto",
        model: Any | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.weights = str(weights)
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            project_root = Path(__file__).resolve().parents[2]
            settings_dir = project_root / "checkpoints" / "ultralytics_config"
            settings_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("YOLO_CONFIG_DIR", str(settings_dir))
            requested_weights = Path(self.weights).expanduser()
            if not requested_weights.is_absolute() and len(requested_weights.parts) == 1 and not requested_weights.is_file():
                weights_dir = project_root / "checkpoints" / "yolo"
                weights_dir.mkdir(parents=True, exist_ok=True)
                requested_weights = weights_dir / requested_weights.name
                self.weights = str(requested_weights)
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("Install ultralytics to use YOLO detection: pip install ultralytics") from exc
            try:
                self._model = YOLO(self.weights)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load YOLO weights {self.weights!r}. Check the path or allow Ultralytics "
                    "to download pretrained COCO weights."
                ) from exc
        return self._model

    def detect_frames(
        self,
        frames_bgr: Sequence[np.ndarray],
        tracking: bool = False,
        persist: bool = True,
    ) -> list[list[ObjectDetection]]:
        """Return detections for each BGR frame with normalized xyxy boxes."""
        if not frames_bgr:
            return []
        if any(frame is None or frame.ndim != 3 or frame.shape[2] != 3 for frame in frames_bgr):
            raise ValueError("Every frame must be a valid BGR image shaped [height, width, 3]")
        selected_device = get_device(self.device)
        yolo_device = "0" if selected_device.type == "cuda" else "cpu"
        if tracking:
            results = self.model.track(
                source=list(frames_bgr), conf=self.confidence_threshold, device=yolo_device,
                persist=persist, stream=False, verbose=False,
            )
        else:
            results = self.model.predict(
                source=list(frames_bgr), conf=self.confidence_threshold, device=yolo_device,
                stream=False, verbose=False,
            )
        detections: list[list[ObjectDetection]] = []
        for frame, result in zip(frames_bgr, results):
            height, width = frame.shape[:2]
            detections.append(parse_ultralytics_result(result, width, height))
        # Ultralytics returns one result per input; fail loudly rather than misalign frames.
        if len(detections) != len(frames_bgr):
            raise RuntimeError(f"YOLO returned {len(detections)} results for {len(frames_bgr)} frames")
        return detections

    def detect_keyframe_paths(
        self, keyframe_paths: Sequence[str | Path], tracking: bool = False,
    ) -> list[list[ObjectDetection]]:
        """Read keyframe image paths and run detection/tracking in temporal order."""
        import cv2

        frames = []
        for path in keyframe_paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Keyframe image is missing or unreadable: {path}")
            frames.append(frame)
        return self.detect_frames(frames, tracking=tracking)


def detections_to_dict(detections: Sequence[ObjectDetection]) -> list[dict[str, Any]]:
    """Convert detection dataclasses to JSON-serializable dictionaries."""
    return [asdict(detection) for detection in detections]

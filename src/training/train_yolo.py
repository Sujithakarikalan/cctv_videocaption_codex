"""Optional YOLOv8 fine-tuning wrapper for validated annotated detection datasets."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.detection_dataset_validator import DetectionDatasetReport, validate_yolo_dataset
from src.utils.device import get_device

LOGGER = logging.getLogger(__name__)


@dataclass
class YOLOTrainingResult:
    run_dir: str
    best_checkpoint: str | None
    last_checkpoint: str | None
    results_csv: str | None
    validation: dict[str, Any]


def _require_trainable_dataset(dataset_yaml: str | Path) -> DetectionDatasetReport:
    report = validate_yolo_dataset(dataset_yaml, splits=("train", "val"))
    errors = [issue for issue in report.issues if issue.severity == "error"]
    if errors:
        sample = "\n".join(f"- {issue.code}: {issue.message} ({issue.path})" for issue in errors[:12])
        raise ValueError(f"YOLO dataset validation failed with {len(errors)} error(s):\n{sample}")
    return report


def train_yolo(
    dataset_yaml: str | Path,
    pretrained_weights: str = "yolov8n.pt",
    epochs: int = 50,
    image_size: int = 640,
    batch_size: int = 16,
    device: str = "auto",
    project: str | Path = "outputs/runs/yolo",
    name: str = "cctv_yolov8",
    seed: int = 42,
    workers: int = 4,
    model: Any | None = None,
) -> YOLOTrainingResult:
    """Fine-tune a pretrained detector; calling this function is explicit opt-in training."""
    if epochs < 1 or image_size < 32 or batch_size < 1 or workers < 0:
        raise ValueError("epochs/batch_size must be positive, image_size >= 32, and workers non-negative")
    report = _require_trainable_dataset(dataset_yaml)
    if model is None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install ultralytics before training: pip install ultralytics") from exc
        try:
            model = YOLO(pretrained_weights)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load starting YOLO weights {pretrained_weights!r}; check the local path "
                "or allow Ultralytics to download pretrained COCO weights."
            ) from exc
    selected_device = get_device(device)
    yolo_device = "0" if selected_device.type == "cuda" else "cpu"
    results = model.train(
        data=str(Path(dataset_yaml).resolve()),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device=yolo_device,
        project=str(Path(project)),
        name=name,
        seed=seed,
        workers=workers,
        pretrained=True,
        exist_ok=True,
        verbose=True,
    )
    save_dir = Path(getattr(results, "save_dir", project / name if isinstance(project, Path) else Path(project) / name))
    weights_dir = save_dir / "weights"
    best_path, last_path, csv_path = weights_dir / "best.pt", weights_dir / "last.pt", save_dir / "results.csv"
    return YOLOTrainingResult(
        run_dir=str(save_dir.resolve()),
        best_checkpoint=str(best_path.resolve()) if best_path.is_file() else None,
        last_checkpoint=str(last_path.resolve()) if last_path.is_file() else None,
        results_csv=str(csv_path.resolve()) if csv_path.is_file() else None,
        validation=report.to_dict(),
    )

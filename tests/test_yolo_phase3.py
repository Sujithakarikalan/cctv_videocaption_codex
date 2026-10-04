"""Phase 3 tests use synthetic YOLO data and a mocked detector; no weights are fetched."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from src.data.detection_dataset_validator import validate_yolo_dataset
from src.features.object_features import aggregate_object_features
from src.models.object_feature_encoder import ObjectFeatureEncoder
from src.models.yolo_extractor import ObjectDetection, YOLODetectionExtractor
from src.training.train_yolo import train_yolo


def make_dataset(root: Path, include_test: bool = True) -> Path:
    splits = ["train", "val", "test"] if include_test else ["train", "val"]
    for split in splits:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 24), "gray").save(image_dir / "frame.png")
        (label_dir / "frame.txt").write_text("0 0.5 0.5 0.4 0.5\n", encoding="utf-8")
    dataset_yaml = root / "dataset.yaml"
    dataset_yaml.write_text(yaml.safe_dump({"path": ".", "nc": 2, "names": ["person", "vehicle"]}), encoding="utf-8")
    return dataset_yaml


def test_yolo_dataset_validates_standard_layout(tmp_path: Path) -> None:
    report = validate_yolo_dataset(make_dataset(tmp_path))
    assert report.is_valid
    assert report.class_count == 2
    assert report.image_count == report.label_count == 3


def test_yolo_validator_reports_missing_malformed_and_mismatched_labels(tmp_path: Path) -> None:
    dataset_yaml = make_dataset(tmp_path)
    (tmp_path / "labels" / "train" / "frame.txt").write_text(
        "4 0.5 0.5 0.3 0.3\n0 1.2 0.5 0.5 0.5\nbad line\n", encoding="utf-8"
    )
    (tmp_path / "labels" / "val" / "orphan.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (tmp_path / "labels" / "test" / "frame.txt").unlink()
    report = validate_yolo_dataset(dataset_yaml)
    codes = {issue.code for issue in report.issues}
    assert not report.is_valid
    assert {"invalid_class_id", "malformed_box", "image_label_mismatch", "missing_label"} <= codes


def test_detection_parser_normalizes_boxes_and_preserves_track_ids() -> None:
    class FakeBoxes:
        xyxy = torch.tensor([[10.0, 5.0, 90.0, 45.0]])
        cls = torch.tensor([0.0])
        conf = torch.tensor([0.95])
        id = torch.tensor([17.0])

        def __len__(self) -> int:
            return 1

    class FakeModel:
        def predict(self, **kwargs):
            assert kwargs["device"] == "cpu"
            return [SimpleNamespace(boxes=FakeBoxes(), names={0: "person"}) for _ in kwargs["source"]]

        def track(self, **kwargs):
            return [SimpleNamespace(boxes=FakeBoxes(), names={0: "person"}) for _ in kwargs["source"]]

    extractor = YOLODetectionExtractor(model=FakeModel(), device="cpu")
    frames = [np.zeros((50, 100, 3), dtype=np.uint8)]
    detections = extractor.detect_frames(frames, tracking=True)
    assert len(detections) == 1 and len(detections[0]) == 1
    detection = detections[0][0]
    assert detection.label == "person"
    assert detection.confidence == pytest.approx(0.95)
    assert detection.bbox_xyxy_normalized == pytest.approx((0.1, 0.1, 0.9, 0.9))
    assert detection.track_id == 17


def test_object_aggregation_includes_counts_confidence_boxes_and_persistence() -> None:
    per_frame = [
        [
            ObjectDetection(0, "person", 0.8, (0.1, 0.2, 0.3, 0.8), 5),
            ObjectDetection(1, "car", 0.7, (0.4, 0.3, 0.9, 0.7), None),
        ],
        [ObjectDetection(0, "person", 0.9, (0.2, 0.2, 0.4, 0.8), 5)],
        [],
    ]
    summary = aggregate_object_features(per_frame)
    person, car = summary.objects
    assert summary.counts_per_frame == [2, 1, 0]
    assert summary.total_detections == 3
    assert summary.mean_objects_per_frame == 1.0
    assert person.persistence == pytest.approx(2 / 3)
    assert person.confidence_mean == pytest.approx(0.85)
    assert person.track_ids == [5]
    assert car.persistence == pytest.approx(1 / 3)
    class_ids, stats, mask = summary.to_tensors()
    assert class_ids.shape == (1, 2)
    assert stats.shape == (1, 2, 8)
    assert mask.all()


def test_object_encoder_handles_populated_and_empty_object_lists() -> None:
    encoder = ObjectFeatureEncoder(num_classes=4, output_dim=16)
    class_ids = torch.tensor([[0, 2], [0, 0]])
    stats = torch.rand(2, 2, 8)
    mask = torch.tensor([[True, True], [False, False]])
    output = encoder(class_ids, stats, mask)
    assert output.shape == (2, 16)
    assert torch.equal(output[1], torch.zeros(16))


def test_training_wrapper_validates_before_calling_model(tmp_path: Path) -> None:
    dataset_yaml = make_dataset(tmp_path / "data", include_test=False)

    class FakeTrainer:
        called = False

        def train(self, **kwargs):
            self.called = True
            assert kwargs["device"] == "cpu"
            return SimpleNamespace(save_dir=tmp_path / "run")

    fake = FakeTrainer()
    result = train_yolo(dataset_yaml, device="cpu", model=fake, project=tmp_path / "runs")
    assert fake.called
    assert result.validation["is_valid"]
    assert result.best_checkpoint is None


def test_training_wrapper_refuses_invalid_dataset_before_model_run(tmp_path: Path) -> None:
    dataset_yaml = make_dataset(tmp_path / "data", include_test=False)
    (tmp_path / "data" / "labels" / "train" / "frame.txt").unlink()

    class FakeTrainer:
        called = False

        def train(self, **kwargs):
            self.called = True

    fake = FakeTrainer()
    with pytest.raises(ValueError, match="dataset validation failed"):
        train_yolo(dataset_yaml, device="cpu", model=fake)
    assert not fake.called

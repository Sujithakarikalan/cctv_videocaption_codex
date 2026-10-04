"""Validation for object-detection datasets in the standard YOLO directory format."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    path: str | None = None


@dataclass
class DetectionDatasetReport:
    dataset_yaml: str
    class_count: int
    image_count: int
    label_count: int
    checked_splits: list[str]
    issues: list[ValidationIssue]

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_yaml": self.dataset_yaml,
            "class_count": self.class_count,
            "image_count": self.image_count,
            "label_count": self.label_count,
            "checked_splits": self.checked_splits,
            "is_valid": self.is_valid,
            "issues": [asdict(issue) for issue in self.issues],
        }


def _resolve_data_root(dataset_yaml: Path, config: dict[str, Any]) -> Path:
    configured = config.get("path")
    if configured is None:
        return dataset_yaml.parent
    root = Path(str(configured)).expanduser()
    return root.resolve() if root.is_absolute() else (dataset_yaml.parent / root).resolve()


def _names_and_count(config: dict[str, Any], issues: list[ValidationIssue], yaml_path: Path) -> tuple[list[str], int]:
    names_value = config.get("names")
    if isinstance(names_value, dict):
        try:
            keys = sorted(int(key) for key in names_value)
            names = [str(names_value[key] if key in names_value else names_value[str(key)]) for key in keys]
            if keys != list(range(len(keys))):
                raise ValueError("class keys must be consecutive integers starting at zero")
        except (TypeError, ValueError, KeyError) as exc:
            issues.append(ValidationIssue("error", "invalid_class_names", f"Invalid YAML class names: {exc}", str(yaml_path)))
            names = []
    elif isinstance(names_value, list):
        names = [str(name) for name in names_value]
    else:
        names = []
        issues.append(ValidationIssue("error", "missing_class_names", "dataset.yaml must define names as a list or integer-keyed mapping", str(yaml_path)))
    declared_count = config.get("nc")
    if declared_count is not None:
        try:
            if int(declared_count) != len(names):
                issues.append(ValidationIssue("error", "class_count_mismatch", f"nc={declared_count} but names contains {len(names)} classes", str(yaml_path)))
        except (TypeError, ValueError):
            issues.append(ValidationIssue("error", "invalid_class_count", f"nc must be an integer, got {declared_count!r}", str(yaml_path)))
    if not names:
        issues.append(ValidationIssue("error", "empty_classes", "At least one class name is required", str(yaml_path)))
    return names, len(names)


def _parse_label_file(label_path: Path, class_count: int, issues: list[ValidationIssue]) -> None:
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        issues.append(ValidationIssue("error", "unreadable_label", str(exc), str(label_path)))
        return
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            issues.append(ValidationIssue(
                "error", "malformed_box", f"Line {line_number}: expected class_id x_center y_center width height; got {len(fields)} fields", str(label_path)
            ))
            continue
        try:
            class_value = float(fields[0])
            box = [float(value) for value in fields[1:]]
            if not class_value.is_integer():
                raise ValueError("class ID is not an integer")
        except ValueError as exc:
            issues.append(ValidationIssue("error", "malformed_box", f"Line {line_number}: non-numeric value ({exc})", str(label_path)))
            continue
        class_id = int(class_value)
        if not 0 <= class_id < class_count:
            issues.append(ValidationIssue("error", "invalid_class_id", f"Line {line_number}: class ID {class_id} outside [0, {class_count - 1}]", str(label_path)))
        x_center, y_center, width, height = box
        if not all(0.0 <= value <= 1.0 for value in box) or width <= 0.0 or height <= 0.0:
            issues.append(ValidationIssue("error", "malformed_box", f"Line {line_number}: normalized box values must be in [0,1] and width/height positive", str(label_path)))
            continue
        if x_center - width / 2 < -1e-6 or x_center + width / 2 > 1 + 1e-6 or y_center - height / 2 < -1e-6 or y_center + height / 2 > 1 + 1e-6:
            issues.append(ValidationIssue("error", "malformed_box", f"Line {line_number}: box extends outside image bounds", str(label_path)))


def validate_yolo_dataset(dataset_yaml: str | Path, splits: tuple[str, ...] = ("train", "val", "test")) -> DetectionDatasetReport:
    """Check YOLO image/label layout, class indices, normalized boxes, and file pairing."""
    yaml_path = Path(dataset_yaml).expanduser().resolve()
    issues: list[ValidationIssue] = []
    if not yaml_path.is_file():
        issues.append(ValidationIssue("error", "missing_dataset_yaml", "dataset.yaml does not exist", str(yaml_path)))
        return DetectionDatasetReport(str(yaml_path), 0, 0, 0, list(splits), issues)
    try:
        config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        issues.append(ValidationIssue("error", "invalid_dataset_yaml", f"Could not read YAML: {exc}", str(yaml_path)))
        return DetectionDatasetReport(str(yaml_path), 0, 0, 0, list(splits), issues)
    if not isinstance(config, dict):
        issues.append(ValidationIssue("error", "invalid_dataset_yaml", "dataset.yaml root must be a mapping", str(yaml_path)))
        return DetectionDatasetReport(str(yaml_path), 0, 0, 0, list(splits), issues)

    _, class_count = _names_and_count(config, issues, yaml_path)
    root = _resolve_data_root(yaml_path, config)
    image_total = 0
    label_total = 0
    for split in splits:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir():
            issues.append(ValidationIssue("error", "missing_image_directory", f"Missing images/{split} directory", str(image_dir)))
            continue
        if not label_dir.is_dir():
            issues.append(ValidationIssue("error", "missing_label_directory", f"Missing labels/{split} directory", str(label_dir)))
            continue
        images = sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        labels = sorted(label_dir.rglob("*.txt"))
        image_total += len(images)
        label_total += len(labels)
        image_by_stem: dict[str, list[Path]] = {}
        for image in images:
            image_by_stem.setdefault(image.stem, []).append(image)
        for image in images:
            label_path = label_dir / image.relative_to(image_dir).with_suffix(".txt")
            if not label_path.is_file():
                issues.append(ValidationIssue("error", "missing_label", f"No matching label file for image {image.name}", str(image)))
        for label in labels:
            relative_label = label.relative_to(label_dir).with_suffix("")
            matching_image = any((image_dir / relative_label).with_suffix(suffix).is_file() for suffix in IMAGE_SUFFIXES)
            if not matching_image:
                issues.append(ValidationIssue("error", "image_label_mismatch", f"Label has no matching image: {label.name}", str(label)))
            _parse_label_file(label, class_count, issues)
        duplicate_stems = [stem for stem, paths in image_by_stem.items() if len(paths) > 1]
        for stem in duplicate_stems:
            issues.append(ValidationIssue("error", "duplicate_image_stem", f"Multiple images share label stem {stem!r}", str(image_dir)))
    return DetectionDatasetReport(str(yaml_path), class_count, image_total, label_total, list(splits), issues)

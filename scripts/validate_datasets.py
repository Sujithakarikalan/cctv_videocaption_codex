"""Validate a YOLO-format detection dataset and print a JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.detection_dataset_validator import validate_yolo_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_yaml", type=Path, help="Path to the dataset.yaml file")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()
    report = validate_yolo_dataset(args.dataset_yaml, tuple(args.splits))
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Fine-tune a pretrained YOLOv8 model on a validated dataset.yaml."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_yolo import train_yolo
from src.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_yaml", type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "yolo_train.yaml")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--project", type=Path, default=None)
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(args.config)
    training = config["training"]
    yolo = config["models"]["yolo"]
    result = train_yolo(
        dataset_yaml=args.dataset_yaml,
        pretrained_weights=args.weights or yolo["weights"],
        epochs=args.epochs or training["epochs"],
        image_size=args.imgsz or yolo["image_size"],
        batch_size=args.batch or training["batch_size"],
        device=args.device or config["models"]["device"],
        project=args.project or PROJECT_ROOT / config["project"]["output_dir"] / "runs" / "yolo",
        name=args.name or yolo["run_name"],
        seed=config["project"]["seed"],
        workers=yolo["workers"],
    )
    print(json.dumps(result.__dict__, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

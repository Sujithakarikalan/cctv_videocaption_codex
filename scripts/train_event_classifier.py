"""Train an optional binary normal/abnormal event classifier from videos."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_event_classifier import train_event_classifier
from src.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="CSV columns video_path,label,split")
    source.add_argument("--directory", type=Path, help="data/events/{train,val,test}/{normal,abnormal}/")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "temporal_train.yaml")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-pretrained", action="store_true", help="Do not request Kinetics pretrained SlowFast weights")
    parser.add_argument("--no-fallback", action="store_true", help="Fail if PyTorchVideo SlowFast is unavailable")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(args.config)
    torch.hub.set_dir(str((PROJECT_ROOT / config["project"]["model_cache_dir"]).resolve()))
    data = config["data"]
    model = config["models"]["temporal"]
    training = config["training"]
    result = train_event_classifier(
        manifest_path=args.manifest,
        directory_root=args.directory,
        output_dir=args.output_dir or PROJECT_ROOT / "checkpoints" / "event_classifier",
        epochs=training["epochs"], batch_size=training["batch_size"],
        learning_rate=training["learning_rate"], weight_decay=training["weight_decay"],
        num_frames=data["num_frames"], sampling_rate=data["sampling_rate"],
        frame_size=data["frame_size"], clips_per_video=data["clips_per_video"],
        alpha=model["alpha"], beta_inv=model["beta_inv"],
        pretrained_temporal=bool(model["pretrained"] and not args.no_pretrained),
        allow_fallback=bool(model["allow_fallback"] and not args.no_fallback),
        fine_tune_temporal=bool(model["fine_tune"]),
        device_name=args.device or config["models"]["device"],
        seed=config["project"]["seed"], workers=training["workers"],
        patience=training["early_stopping_patience"],
        gradient_clip_norm=training["gradient_clip_norm"],
        mixed_precision=training["mixed_precision"],
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

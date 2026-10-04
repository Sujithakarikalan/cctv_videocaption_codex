"""Run pretrained YOLOv8 on keyframes and save detections plus object-encoder input tensors."""

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

from src.features.object_features import aggregate_object_features
from src.models.yolo_extractor import YOLODetectionExtractor, detections_to_dict
from src.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="Keyframe image paths in temporal order")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "base.yaml")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--track", action="store_true", help="Enable optional Ultralytics object tracking")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "predictions" / "detections.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(args.config)
    model_config = config["models"]["yolo"]
    extractor = YOLODetectionExtractor(
        weights=args.weights or model_config["weights"],
        confidence_threshold=args.confidence if args.confidence is not None else model_config["confidence_threshold"],
        device=args.device or config["models"]["device"],
    )
    per_frame = extractor.detect_keyframe_paths(args.images, tracking=args.track)
    summary = aggregate_object_features(per_frame)
    class_ids, object_stats, object_mask = summary.to_tensors()
    feature_path = args.output.with_suffix(".pt")
    result = {
        "weights": args.weights or model_config["weights"],
        "tracking_enabled": args.track,
        "frames": [
            {"path": str(path.resolve()), "detections": detections_to_dict(detections)}
            for path, detections in zip(args.images, per_frame)
        ],
        "object_summary": summary.to_dict(),
        "next_stage_features": {
            "format": "ObjectFeatureEncoder inputs",
            "class_ids_shape": list(class_ids.shape),
            "stats_shape": list(object_stats.shape),
            "stats_columns": [
                "count_per_frame", "confidence_mean", "confidence_max", "center_x", "center_y",
                "box_width", "box_height", "temporal_persistence",
            ],
            "mask_shape": list(object_mask.shape),
            "tensor_path": str(feature_path.resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({
        "schema_version": 1,
        "source_detection_json": str(args.output.resolve()),
        "class_ids": class_ids,
        "stats": object_stats,
        "object_mask": object_mask,
    }, feature_path)
    print(f"Frames processed: {len(per_frame)}")
    print(f"Total detections: {summary.total_detections}")
    print(f"Object summary: {args.output.resolve()}")
    print(f"Next-stage object features: {feature_path.resolve()} (stats shape: {tuple(object_stats.shape)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

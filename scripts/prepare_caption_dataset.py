"""Validate caption-reference CSVs and create source-level data splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.caption_dataset import split_caption_manifest, validate_caption_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="Validate a caption manifest")
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--require-splits", action="store_true")
    split = subparsers.add_parser("split", help="Assign source groups to train/validation/test")
    split.add_argument("manifest", type=Path)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--train-ratio", type=float, default=0.8)
    split.add_argument("--validation-ratio", type=float, default=0.1)
    split.add_argument("--test-ratio", type=float, default=0.1)
    split.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        if args.command == "validate":
            report = validate_caption_manifest(args.manifest, require_splits=args.require_splits)
            print(json.dumps(report, indent=2))
            return 0 if report["valid"] else 1
        counts = split_caption_manifest(
            args.manifest, args.output,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        report = validate_caption_manifest(args.output, require_splits=True)
        print(json.dumps({"output": str(args.output.resolve()), "video_counts": counts, **report}, indent=2))
        return 0 if report["valid"] else 1
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

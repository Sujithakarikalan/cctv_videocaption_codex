"""Extract motion-aware ResNet18 semantic keyframes from one video."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.semantic_keyframes import extract_semantic_keyframes
from src.models.resnet18_encoder import ResNet18Encoder
from src.utils.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Input MP4/AVI path")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "base.yaml")
    parser.add_argument(
        "--no-pretrained", action="store_true",
        help="Use random ResNet18 weights for a smoke test only (not valid for semantic selection)",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(args.config)
    data_config = config.get("data", {})
    encoder_config = config.get("visual_encoder", {})
    root = PROJECT_ROOT
    output_dir = args.output_dir or root / data_config.get("processed_dir", "data/processed") / "keyframes"
    cache_dir = args.cache_dir or root / data_config.get("feature_dir", "data/features") / "resnet18"
    model_cache_dir = root / config.get("project", {}).get("model_cache_dir", "checkpoints/model_cache")
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(model_cache_dir.resolve()))
    pretrained = not args.no_pretrained and bool(encoder_config.get("pretrained", True))
    if args.no_pretrained:
        logging.warning(
            "--no-pretrained uses random, untrained ResNet18 features; output is only a smoke test, "
            "not valid for real semantic keyframe selection."
        )
    encoder = ResNet18Encoder(pretrained=pretrained, freeze_backbone=True)
    selection = extract_semantic_keyframes(
        video_path=args.video,
        output_dir=output_dir,
        cache_dir=cache_dir,
        encoder=encoder,
        device=args.device or encoder_config.get("device", config.get("models", {}).get("device", "auto")),
        sample_fps=float(data_config.get("sample_fps", 2.0)),
        frame_size=int(data_config.get("frame_size", 224)),
        max_keyframes=int(data_config.get("max_keyframes", 8)),
        motion_threshold=float(data_config.get("motion_threshold", 0.015)),
        semantic_similarity_threshold=float(data_config.get("semantic_similarity_threshold", 0.88)),
        min_gap_seconds=float(data_config.get("minimum_keyframe_gap_seconds", 0.5)),
    )
    print(f"Video ID: {selection.video_id}")
    print(f"Duration: {selection.metadata['duration_seconds']:.3f}s")
    print(f"Selected keyframes: {len(selection.selected_frames)}")
    print(f"Feature shape: {tuple(selection.features.shape)}")
    print(f"Metadata: {selection.metadata_path}")
    print(f"Contact sheet: {selection.contact_sheet_path}")
    print(f"Feature cache: {selection.cache_path} (cache hit: {selection.cache_hit})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

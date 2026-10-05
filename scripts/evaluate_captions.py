"""Run Phase 6 Qwen3-VL captions on test videos and score human references."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.caption_dataset import load_caption_rows, validate_caption_manifest
from src.data.video_io import get_video_metadata
from src.evaluation.caption_evaluation import evaluate_caption_rows
from src.models.qwen3_video_captioner import MODEL_NAME, Qwen3VideoCaptioner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="CSV manifest containing human references and split labels")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--max-pixels", type=int, default=151200)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "checkpoints" / "qwen3_vl")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON path (default outputs/predictions/caption_evaluation.json)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if min(args.num_frames, args.max_pixels, args.max_new_tokens) < 1:
        parser.error("--num-frames, --max-pixels, and --max-new-tokens must be positive")
    report = validate_caption_manifest(args.manifest, require_splits=True)
    if not report["valid"]:
        logging.error("Invalid caption manifest:\n- %s", "\n- ".join(report["errors"]))
        return 2
    try:
        rows = load_caption_rows(args.manifest, split="test")
        if not rows:
            raise ValueError("Manifest has no test rows; create a source-level test split first")
        model_load_started = time.perf_counter()
        captioner = Qwen3VideoCaptioner(device=args.device, cache_dir=args.cache_dir)
        model_load_seconds = time.perf_counter() - model_load_started

        def generate(video_path: str) -> Any:
            metadata = get_video_metadata(video_path)
            frames = min(args.num_frames, metadata.total_frames) if metadata.total_frames > 0 else args.num_frames
            return captioner.caption(
                video_path,
                num_frames=frames,
                max_pixels=args.max_pixels,
                max_new_tokens=args.max_new_tokens,
                supporting_text="",
            )

        evaluation_started = time.perf_counter()
        evaluated = evaluate_caption_rows(rows, generate)
        evaluation_seconds = time.perf_counter() - evaluation_started
    except Exception as exc:
        logging.error("Caption evaluation failed: %s", exc)
        return 2

    output = args.output or PROJECT_ROOT / "outputs" / "predictions" / "caption_evaluation.json"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    devices = sorted({p["device"] for p in evaluated["predictions"] if p["device"]})
    payload = {
        "schema_version": 1,
        "task": "held-out CCTV caption evaluation against human-written references",
        "model": MODEL_NAME,
        "device": devices[0] if len(devices) == 1 else devices or args.device,
        "manifest": str(args.manifest.expanduser().resolve()),
        "split_evaluated": "test",
        "sampling": {"strategy": "uniform temporal sampling by Qwen3-VL video processor",
                     "requested_num_frames": args.num_frames, "max_pixels_per_frame": args.max_pixels},
        "timing_seconds": {"model_load": round(model_load_seconds, 3),
                           "evaluation_inference": round(evaluation_seconds, 3)},
        "metrics": evaluated["metrics"],
        "metric_notes": {
            "corpus_bleu_4": "Smoothed corpus BLEU-4, range 0-1; lexical overlap and sensitive to wording.",
            "mean_best_reference_rouge_l_f1": "Mean per-video ROUGE-L F1 against its best human reference, range 0-1.",
            "small_dataset": "With a small CCTV set, scores are descriptive only; inspect every saved example.",
        },
        "examples_for_human_review": evaluated["predictions"],
        "human_references_required": True,
        "supporting_models_used": {"yolo": False, "slowfast": False},
        "output_path": str(output),
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"metrics": payload["metrics"], "output": str(output), "device": payload["device"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

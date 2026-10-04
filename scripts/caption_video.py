"""Generate a factual CCTV caption with pretrained Qwen3-VL-4B-Instruct."""

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

from src.data.video_io import get_video_metadata
from src.features.caption_evidence import (
    find_keyframe_manifest,
    find_phase5_results,
    format_caption_support,
    matches_video_path,
    select_slowfast_hints,
    summarize_yolo_hints,
)
from src.models.qwen3_video_captioner import MODEL_NAME, Qwen3VideoCaptioner


def _load_evidence(video_path: Path, phase5_path: Path | None, use_yolo: bool) -> tuple[
    list[dict[str, Any]], dict[str, Any], dict[str, Any] | None, str | None
]:
    phase5_result: dict[str, Any] | None = None
    phase5_source: str | None = None
    if phase5_path is not None:
        phase5_source = str(phase5_path.expanduser().resolve())
        phase5_result = json.loads(phase5_path.read_text(encoding="utf-8"))
        result_video = phase5_result.get("video", {}).get("path")
        if not matches_video_path(result_video, video_path):
            raise ValueError(f"Phase 5 results do not match the requested video: {phase5_path}")
    else:
        match = find_phase5_results(video_path, PROJECT_ROOT)
        if match is not None:
            phase5_path, phase5_result = match
            phase5_source = str(phase5_path.resolve())

    frame_records: list[dict[str, Any]] = []
    source = "not available"
    if use_yolo and phase5_result is not None:
        frame_records = list(phase5_result.get("yolo_keyframes", []))
        source = "saved Phase 5 output" if frame_records else "Phase 5 output has no YOLO keyframes"
    elif use_yolo:
        manifest_match = find_keyframe_manifest(video_path, PROJECT_ROOT)
        if manifest_match is not None:
            manifest_path, manifest = manifest_match
            from src.models.yolo_extractor import YOLODetectionExtractor, detections_to_dict

            items = manifest.get("keyframes", [])
            paths = []
            for item in items:
                path = Path(item["path"])
                if not path.is_file():
                    path = manifest_path.parent / path.name
                paths.append(path)
            extractor = YOLODetectionExtractor(device="auto")
            per_frame = extractor.detect_keyframe_paths(paths)
            frame_records = [
                {"timestamp_seconds": float(item.get("timestamp_seconds", 0.0)),
                 "path": str(path.resolve()), "detections": detections_to_dict(detections)}
                for item, path, detections in zip(items, paths, per_frame)
            ]
            source = f"YOLO run on Phase 2 keyframes: {manifest_path.resolve()}"
        else:
            source = "no saved Phase 5 output or matching Phase 2 keyframe manifest"

    hints, yolo_metadata = summarize_yolo_hints(frame_records)
    yolo_metadata["source"] = source
    yolo_metadata["enabled"] = use_yolo
    return hints if use_yolo else [], yolo_metadata, phase5_result, phase5_source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Input CCTV video file")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--num-frames", type=int, default=16,
                        help="Uniformly sampled frames sent to Qwen (default: 16)")
    parser.add_argument("--max-pixels", type=int, default=151200,
                        help="Per-frame Qwen pixel budget; lower this to reduce GPU memory use")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--no-yolo-hints", action="store_true",
                        help="Do not load cached YOLO detections or run YOLO on Phase 2 keyframes")
    parser.add_argument("--phase5-results", type=Path, default=None,
                        help="Optional saved Phase 5 JSON for YOLO and SlowFast evidence")
    parser.add_argument("--use-slowfast-hints", action="store_true",
                        help="Pass top-1-per-clip SlowFast labels as explicitly low-trust hints")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "checkpoints" / "qwen3_vl")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        parser.error(f"Video file not found: {video_path}")
    if args.num_frames < 1 or args.max_pixels < 1 or args.max_new_tokens < 1:
        parser.error("--num-frames, --max-pixels, and --max-new-tokens must be positive")
    try:
        metadata = get_video_metadata(video_path)
        yolo_hints, yolo_metadata, phase5_result, phase5_source = _load_evidence(
            video_path, args.phase5_results, use_yolo=not args.no_yolo_hints
        )
        slowfast_hints, slowfast_metadata = select_slowfast_hints(
            phase5_result, enabled=args.use_slowfast_hints
        )
        support_text = format_caption_support(yolo_hints, slowfast_hints)

        requested_frames = args.num_frames
        effective_frames = min(requested_frames, metadata.total_frames) if metadata.total_frames > 0 else requested_frames
        started = time.perf_counter()
        load_started = time.perf_counter()
        captioner = Qwen3VideoCaptioner(device=args.device, cache_dir=args.cache_dir)
        model_load_seconds = time.perf_counter() - load_started
        generation = captioner.caption(
            video_path,
            num_frames=effective_frames,
            max_pixels=args.max_pixels,
            max_new_tokens=args.max_new_tokens,
            supporting_text=support_text,
        )
        total_seconds = time.perf_counter() - started
    except Exception as exc:
        logging.error("Caption inference failed: %s", exc)
        return 2

    sample_indices: list[int] = []
    sample_timestamps: list[float] = []
    if metadata.total_frames > 0 and effective_frames > 0:
        import numpy as np

        sample_indices = np.linspace(0, metadata.total_frames - 1, effective_frames, dtype=np.int64).tolist()
        sample_timestamps = [round(index / metadata.fps, 4) for index in sample_indices]

    output_path = args.output or PROJECT_ROOT / "outputs" / "predictions" / f"{video_path.stem}_caption.json"
    result = {
        "schema_version": 1,
        "video": {
            "path": str(video_path), "name": video_path.name,
            "fps": metadata.fps, "duration_seconds": metadata.duration_seconds,
            "width": metadata.width, "height": metadata.height, "total_frames": metadata.total_frames,
        },
        "caption": generation.caption,
        "model": generation.model_name,
        "device": generation.device,
        "timing_seconds": {
            "model_load": round(model_load_seconds, 3),
            "caption_inference": round(generation.inference_seconds, 3),
            "total_after_evidence": round(total_seconds, 3),
        },
        "sampling": {
            "strategy": "Qwen3-VL native video processor with uniform num_frames sampling; fps=None",
            "requested_num_frames": requested_frames,
            "num_frames_sent": effective_frames,
            "approx_sample_frame_indices": sample_indices,
            "approx_sample_timestamps_seconds": sample_timestamps,
            "max_pixels_per_frame": args.max_pixels,
        },
        "yolo_support": {
            **yolo_metadata,
            "hints_provided_to_model": yolo_hints,
            "interpretation": "filtered detector evidence only; repeated frame detections are not unique-object counts",
        },
        "slowfast_support": {
            **slowfast_metadata,
            "phase5_source": phase5_source,
            "interpretation": "optional, low-trust Kinetics suggestions; scores are not incident probabilities; internal use cannot be verified",
        },
        "output_path": str(output_path.resolve()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(generation.caption)
    print(f"Device: {generation.device}; inference: {generation.inference_seconds:.2f}s")
    print(f"Caption result: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

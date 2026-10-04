"""Run pretrained YOLO on Phase 2 keyframes and Kinetics SlowFast on real video clips."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.object_features import aggregate_object_features
from src.models.slowfast_action_recognizer import SlowFastActionRecognizer, predictions_to_dict
from src.models.yolo_extractor import YOLODetectionExtractor, detections_to_dict
from src.utils.device import get_device
from src.data.video_io import get_video_metadata


def find_keyframe_manifest(video_path: Path, explicit: Path | None) -> dict | None:
    if explicit is not None:
        candidates = [explicit]
    else:
        candidates = sorted((PROJECT_ROOT / "data" / "processed" / "keyframes").glob("*/keyframes.json"))
    target = video_path.resolve()
    for candidate in candidates:
        if not candidate.is_file():
            continue
        manifest = json.loads(candidate.read_text(encoding="utf-8"))
        if Path(manifest.get("video", {}).get("path", "")).resolve() == target:
            manifest["_manifest_path"] = str(candidate.resolve())
            return manifest
    if explicit:
        raise ValueError(f"Keyframe manifest does not refer to {video_path}: {explicit}")
    return None


def decode_clip(video_path: Path, start: float, duration: float, frame_count: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    frames: list[np.ndarray] = []
    try:
        times = np.linspace(start, start + duration, num=frame_count, endpoint=False)
        for timestamp in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp * 1000))
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"Could not decode clip at {start:.3f}s from {video_path}")
    # A very short clip is repeated to the required temporal length; normal clips already have 32 frames.
    while len(frames) < frame_count:
        frames.append(frames[-1].copy())
    return frames


def choose_clip_starts(duration: float, clip_duration: float, max_clips: int) -> list[float]:
    last_start = max(0.0, duration - clip_duration)
    if max_clips == 1 or last_start == 0:
        return [0.0]
    return [float(value) for value in np.linspace(0.0, last_start, num=max_clips)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Input MP4/video path")
    parser.add_argument("--keyframes-manifest", type=Path, default=None,
                        help="Phase 2 keyframes.json (auto-matched when omitted)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-clips", type=int, default=3,
                        help="Number of uniformly-spaced 2.13-second SlowFast clips")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        parser.error(f"Video file not found: {video_path}")
    if args.top_k < 1 or args.max_clips < 1:
        parser.error("--top-k and --max-clips must be positive")

    metadata = get_video_metadata(video_path)
    selected_device = get_device(args.device)
    manifest = find_keyframe_manifest(video_path, args.keyframes_manifest)
    yolo = YOLODetectionExtractor(device=str(selected_device))
    if manifest and manifest.get("keyframes"):
        keyframes = manifest["keyframes"]
        paths = [Path(item["path"]) for item in keyframes]
        detections = yolo.detect_keyframe_paths(paths)
        object_summary = aggregate_object_features(detections)
        yolo_records = [
            {"timestamp_seconds": float(item["timestamp_seconds"]), "path": str(path),
             "detections": detections_to_dict(frame_detections)}
            for item, path, frame_detections in zip(keyframes, paths, detections)
        ]
        keyframe_source = manifest["_manifest_path"]
    else:
        manifest = None
        keyframe_source = None
        object_summary = aggregate_object_features([])
        yolo_records = []

    recognizer = SlowFastActionRecognizer(device=args.device)
    clip_duration = (32 * 2) / 30.0  # Official hub transform: 32 frames at sampling rate 2, 30 FPS.
    starts = choose_clip_starts(metadata.duration_seconds, clip_duration, args.max_clips)
    action_clips = []
    for start in starts:
        frames = decode_clip(video_path, start, min(clip_duration, max(metadata.duration_seconds - start, 0.05)), 32)
        predictions = recognizer.predict_frames(frames, top_k=args.top_k)
        action_clips.append({
            "start_seconds": start,
            "end_seconds": min(start + clip_duration, metadata.duration_seconds),
            "predictions": predictions_to_dict(predictions),
        })

    class_ids, object_stats, object_mask = object_summary.to_tensors()
    output = args.output or PROJECT_ROOT / "outputs" / "predictions" / f"{video_path.stem}_phase5_actions.json"
    tensor_path = output.with_suffix(".pt")
    result = {
        "schema_version": 1,
        "video": {"path": str(video_path), "fps": metadata.fps, "frames": metadata.total_frames,
                  "duration_seconds": metadata.duration_seconds, "width": metadata.width, "height": metadata.height},
        "models": {
            "object_detection": {"name": "YOLOv8n", "weights": str(PROJECT_ROOT / "checkpoints" / "yolo" / "yolov8n.pt"),
                                 "source": "Phase 2 keyframes" if manifest else "not run (no Phase 2 manifest found)"},
            "action_recognition": {"name": recognizer.backend, "weights": "Kinetics-400 pretrained SlowFast-R50 (official PyTorchVideo)",
                                   "device": str(recognizer.device), "labels_count": len(recognizer.labels),
                                   "input_frames": 32, "clip_sampling_rate": 2, "clip_duration_seconds": clip_duration},
        },
        "keyframe_manifest": keyframe_source,
        "yolo_keyframes": yolo_records,
        "object_summary": object_summary.to_dict(),
        "slowfast_action_clips": action_clips,
        "interpretation_note": "Kinetics-400 class probabilities are clip classifications, not calibrated CCTV incident probabilities. YOLO objects are contextual outputs; the pretrained SlowFast head is not conditioned on YOLO features.",
        "next_stage_object_tensors": {"path": str(tensor_path.resolve()), "class_ids_shape": list(class_ids.shape),
                                       "stats_shape": list(object_stats.shape), "mask_shape": list(object_mask.shape),
                                       "stats_columns": ["count_per_frame", "confidence_mean", "confidence_max", "center_x", "center_y", "box_width", "box_height", "temporal_persistence"]},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({"schema_version": 1, "video_path": str(video_path), "class_ids": class_ids,
                "stats": object_stats, "object_mask": object_mask}, tensor_path)
    print(f"YOLO keyframes processed: {len(yolo_records)}; objects: {object_summary.total_detections}")
    print(f"Object features for downstream use: {tensor_path} (stats={tuple(object_stats.shape)})")
    for clip in action_clips:
        print(f"\nClip {clip['start_seconds']:.2f}-{clip['end_seconds']:.2f}s")
        for prediction in clip["predictions"]:
            print(f"  {prediction['label']}: {prediction['confidence']:.4f}")
    print(f"\nFull results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

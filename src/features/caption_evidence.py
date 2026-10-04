"""Discover and filter optional YOLO/SlowFast evidence for video caption prompts."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


def matches_video_path(recorded_path: str | Path | None, video_path: str | Path) -> bool:
    """Match exact paths, or portable basenames when a project moves between hosts."""
    if not recorded_path:
        return False
    recorded = Path(recorded_path).expanduser()
    target = Path(video_path).expanduser()
    try:
        if recorded.resolve() == target.resolve():
            return True
    except OSError:
        pass
    return recorded.name.casefold() == target.name.casefold()


def find_phase5_results(video_path: str | Path, project_root: str | Path) -> tuple[Path, dict[str, Any]] | None:
    """Find a saved Phase 5 result matching the resolved source-video path."""
    root = Path(project_root) / "outputs" / "predictions"
    for path in sorted(root.glob("*_phase5_actions.json")):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            source = result.get("video", {}).get("path")
            if matches_video_path(source, video_path):
                return path, result
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
    return None


def find_keyframe_manifest(video_path: str | Path, project_root: str | Path) -> tuple[Path, dict[str, Any]] | None:
    root = Path(project_root) / "data" / "processed" / "keyframes"
    for path in sorted(root.glob("*/keyframes.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            source = manifest.get("video", {}).get("path")
            if matches_video_path(source, video_path):
                return path, manifest
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
    return None


def summarize_yolo_hints(
    frame_records: Sequence[dict[str, Any]], min_confidence: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Summarize confident classes without presenting frame hits as unique object counts."""
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"scores": [], "frames": set(), "track_ids": set()})
    total = excluded = 0
    for frame_index, frame in enumerate(frame_records):
        for detection in frame.get("detections", []):
            total += 1
            score = float(detection.get("confidence", 0.0))
            if score < min_confidence:
                excluded += 1
                continue
            label = str(detection.get("label", "unknown"))
            item = grouped[label]
            item["scores"].append(score)
            item["frames"].add(frame_index)
            track_id = detection.get("track_id")
            if track_id is not None:
                item["track_ids"].add(int(track_id))

    hints: list[dict[str, Any]] = []
    frame_count = len(frame_records)
    for label, item in sorted(grouped.items(), key=lambda pair: max(pair[1]["scores"]), reverse=True):
        scores = item["scores"]
        hints.append({
            "label": label,
            "detection_hits_across_frames": len(scores),
            "sampled_frames_present": len(item["frames"]),
            "sampled_frame_count": frame_count,
            "mean_confidence": sum(scores) / len(scores),
            "max_confidence": max(scores),
            "tracked_ids": sorted(item["track_ids"]),
            "unique_object_count": len(item["track_ids"]) if item["track_ids"] else None,
            "interpretation": (
                "unique tracked IDs" if item["track_ids"] else
                "frame-level detections; not a unique-object count"
            ),
        })
    metadata = {
        "sampled_frames": frame_count,
        "detections_total": total,
        "detections_below_confidence_threshold": excluded,
        "minimum_confidence": min_confidence,
        "unique_counts_available": any(item["track_ids"] for item in grouped.values()),
    }
    return hints, metadata


def select_slowfast_hints(
    result: dict[str, Any] | None,
    enabled: bool,
    max_hints: int = 6,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select action suggestions only by explicit opt-in; scores remain Kinetics class scores."""
    available = bool(result and result.get("slowfast_action_clips"))
    if not enabled or not available:
        return [], {
            "available": available,
            "enabled": enabled,
            "provided_to_model": [],
            "excluded_by_policy": [],
            "usage_status": "not requested" if not enabled else "no saved Phase 5 action hints found",
            "score_interpretation": "Kinetics class score only; never abnormality or incident probability",
        }
    candidates: list[dict[str, Any]] = []
    for clip in result.get("slowfast_action_clips", []):
        for prediction in clip.get("predictions", [])[:1]:
            candidates.append({
                "start_seconds": float(clip.get("start_seconds", 0.0)),
                "end_seconds": float(clip.get("end_seconds", 0.0)),
                "label": str(prediction.get("label", "unknown")),
                "kinetics_score": float(prediction.get("confidence", 0.0)),
            })
    provided = candidates[:max_hints]
    excluded = candidates[max_hints:]
    return provided, {
        "available": True,
        "enabled": True,
        "provided_to_model": provided,
        "excluded_by_policy": excluded,
        "usage_status": "passed as low-trust suggestions; internal model use cannot be verified",
        "score_interpretation": "Kinetics class score only; never abnormality or incident probability",
    }


def format_caption_support(yolo_hints: Sequence[dict[str, Any]], slowfast_hints: Sequence[dict[str, Any]]) -> str:
    """Format concise evidence for the VLM without claiming repeated boxes are distinct objects."""
    sections: list[str] = []
    if yolo_hints:
        object_lines = []
        for hint in yolo_hints:
            presence = f"seen in {hint['sampled_frames_present']}/{hint['sampled_frame_count']} sampled frames"
            object_lines.append(
                f"- {hint['label']}: {presence}; max detector confidence {hint['max_confidence']:.2f}; "
                f"{hint['interpretation']}"
            )
        sections.append("YOLO object hints (not guaranteed; do not infer unique counts):\n" + "\n".join(object_lines))
    if slowfast_hints:
        action_lines = [
            f"- {hint['start_seconds']:.1f}-{hint['end_seconds']:.1f}s: {hint['label']} "
            f"(Kinetics class score {hint['kinetics_score']:.3f}; weak, possibly incorrect suggestion)"
            for hint in slowfast_hints
        ]
        sections.append("SlowFast action suggestions (low trust; ignore if the frames disagree):\n" + "\n".join(action_lines))
    return "\n\n".join(sections)

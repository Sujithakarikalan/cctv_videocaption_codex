"""Motion-aware ResNet18 semantic keyframe selection and visual feature caching."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import torch
from PIL import Image, ImageDraw

from src.data.motion_filter import filter_by_motion
from src.data.video_io import SampledFrame, get_video_metadata, iter_sampled_frames, metadata_to_dict
from src.features.feature_cache import FeatureCache, file_sha256, write_json
from src.models.resnet18_encoder import ResNet18Encoder, encode_bgr_frames
from src.utils.device import get_device


@dataclass
class KeyframeSelection:
    video_id: str
    metadata: dict[str, Any]
    selected_frames: list[SampledFrame]
    features: torch.Tensor
    motion_scores: list[float]
    keyframe_paths: list[str]
    cache_path: str
    metadata_path: str
    contact_sheet_path: str
    cache_hit: bool


def _video_id(path: Path) -> str:
    return f"{path.stem}_{hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:10]}"


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return cosine similarity for two same-shaped vectors, clamped to [-1, 1]."""
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("cosine_similarity expects same-shaped one-dimensional feature vectors")
    value = torch.nn.functional.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0), dim=1)[0]
    return float(value.clamp(-1.0, 1.0).item())


def _select_by_similarity(
    features: torch.Tensor,
    frames: list[SampledFrame],
    motion_scores_for_frames: list[float],
    max_keyframes: int,
    semantic_similarity_threshold: float,
    motion_threshold: float,
    min_gap_seconds: float,
) -> list[int]:
    """Retain first/last and candidates with motion OR sufficient semantic novelty."""
    if not frames:
        return []
    if max_keyframes < 1:
        raise ValueError("max_keyframes must be at least 1")
    if len(frames) == 1:
        return [0]
    if max_keyframes < 2:
        raise ValueError("max_keyframes must be at least 2 to preserve both endpoints")
    if not 0.0 <= semantic_similarity_threshold <= 1.0:
        raise ValueError("semantic_similarity_threshold must be between 0 and 1")

    chosen = [0]
    previous = 0
    for index in range(1, len(frames) - 1):
        similarity = cosine_similarity(features[index], features[previous])
        gap_ok = frames[index].timestamp_seconds - frames[previous].timestamp_seconds >= min_gap_seconds
        motion_event = motion_scores_for_frames[index] >= motion_threshold
        semantic_change = gap_ok and similarity < semantic_similarity_threshold
        if (motion_event or semantic_change) and len(chosen) < max_keyframes - 1:
            chosen.append(index)
            previous = index
    chosen.append(len(frames) - 1)
    return chosen


def _save_contact_sheet(paths: list[str], destination: Path, thumbnail_size: tuple[int, int] = (240, 150)) -> None:
    if not paths:
        raise ValueError("Cannot create contact sheet without keyframes")
    images = [Image.open(path).convert("RGB") for path in paths]
    columns = min(4, len(images))
    rows = (len(images) + columns - 1) // columns
    label_height = 24
    sheet = Image.new("RGB", (columns * thumbnail_size[0], rows * (thumbnail_size[1] + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, image in enumerate(images):
        x = (index % columns) * thumbnail_size[0]
        y = (index // columns) * (thumbnail_size[1] + label_height)
        image.thumbnail(thumbnail_size)
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + thumbnail_size[1] + 3), f"Keyframe {index + 1}", fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def _write_keyframe_metadata(
    destination: Path,
    video_id: str,
    metadata: dict[str, Any],
    cache_path: Path,
    cache_parameters: dict[str, Any],
    frames: list[SampledFrame],
    motion_scores_for_frames: list[float],
    keyframe_paths: list[str],
    contact_sheet_path: Path,
) -> None:
    write_json(destination, {
        "video_id": video_id,
        "video": metadata,
        "encoder_name": "resnet18",
        "feature_dimension": 512,
        "pretrained": cache_parameters["pretrained"],
        "weights": cache_parameters["weights"],
        "preprocessing": cache_parameters["preprocessing"],
        "selection_parameters": cache_parameters,
        "cache_path": str(cache_path.resolve()),
        "cache_schema_version": 2,
        "keyframes": [
            {"path": keyframe_paths[i], "frame_index": frame.frame_index,
             "timestamp_seconds": frame.timestamp_seconds, "motion_score": motion_scores_for_frames[i]}
            for i, frame in enumerate(frames)
        ],
        "contact_sheet": str(contact_sheet_path.resolve()),
    })


def extract_semantic_keyframes(
    video_path: str | Path,
    output_dir: str | Path,
    cache_dir: str | Path,
    encoder: ResNet18Encoder | None = None,
    device: str = "auto",
    sample_fps: float = 2.0,
    frame_size: int = 224,
    max_keyframes: int = 8,
    motion_threshold: float = 0.015,
    semantic_similarity_threshold: float = 0.88,
    min_gap_seconds: float = 0.5,
    batch_size: int = 16,
) -> KeyframeSelection:
    """Extract motion candidates, select semantic keyframes, and cache ResNet18 features."""
    path = Path(video_path).expanduser().resolve()
    video_metadata = get_video_metadata(path)
    if encoder is None:
        encoder = ResNet18Encoder(pretrained=True, freeze_backbone=True)
    preprocessing = {
        "color": "RGB", "resize": [frame_size, frame_size],
        "interpolation": "opencv.INTER_AREA", "scale": "uint8_to_float_0_1",
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        "normalization": "L2",
    }
    cache_parameters = {
        "encoder_name": "resnet18", "feature_dimension": 512,
        "pretrained": bool(encoder.pretrained), "weights": encoder.weights_name,
        "weights_url": encoder.weights_url, "preprocessing": preprocessing,
        "sample_fps": float(sample_fps), "frame_size": int(frame_size),
        "max_keyframes": int(max_keyframes), "motion_threshold": float(motion_threshold),
        "semantic_similarity_threshold": float(semantic_similarity_threshold),
        "minimum_keyframe_gap_seconds": float(min_gap_seconds),
    }
    video_id = _video_id(path)
    cache = FeatureCache(cache_dir)
    source_hash = file_sha256(path)
    cache_entry = cache.load(video_id, source_hash, cache_parameters)
    frame_root = Path(output_dir) / video_id
    metadata_path = frame_root / "keyframes.json"
    contact_path = frame_root / "contact_sheet.jpg"

    if cache_entry is not None:
        cached = cache_entry["features"]
        indices = cached["frame_indices"].tolist()
        timestamps = cached["timestamps"].tolist()
        motion_values = cached["motion_scores"].tolist()
        paths = cached["keyframe_paths"]
        frames = [
            SampledFrame(int(frame_index), float(timestamp), cv2.imread(str(saved_path), cv2.IMREAD_COLOR))
            for frame_index, timestamp, saved_path in zip(indices, timestamps, paths)
        ]
        if any(frame.frame_bgr is None for frame in frames):
            raise ValueError("Cached keyframe image is missing or unreadable; remove cache and rerun extraction")
        if not contact_path.is_file():
            _save_contact_sheet(paths, contact_path)
        _write_keyframe_metadata(
            metadata_path, video_id, metadata_to_dict(video_metadata), cache.path_for(video_id),
            cache_parameters, frames, motion_values, paths, contact_path,
        )
        return KeyframeSelection(
            video_id, metadata_to_dict(video_metadata), frames, cached["visual_embeddings"],
            motion_values, paths, str(cache.path_for(video_id).resolve()), str(metadata_path.resolve()),
            str(contact_path.resolve()), True,
        )

    raw_frames = list(iter_sampled_frames(path, sample_fps=sample_fps))
    filtered = filter_by_motion(raw_frames, threshold=motion_threshold)
    if not filtered.frames:
        raise ValueError(f"No usable frames were decoded from {path}")
    candidate_features = encode_bgr_frames(
        encoder, [frame.frame_bgr for frame in filtered.frames], get_device(device),
        frame_size=frame_size, batch_size=batch_size,
    )
    candidate_features = torch.nn.functional.normalize(candidate_features, p=2, dim=1)
    chosen_positions = _select_by_similarity(
        candidate_features, filtered.frames, filtered.scores, max_keyframes,
        semantic_similarity_threshold, motion_threshold, min_gap_seconds,
    )
    selected_frames = [filtered.frames[index] for index in chosen_positions]
    selected_features = candidate_features[chosen_positions].cpu()
    selected_motion = [filtered.scores[index] for index in chosen_positions]

    frame_root.mkdir(parents=True, exist_ok=True)
    keyframe_paths: list[str] = []
    for position, frame in enumerate(selected_frames):
        destination = frame_root / f"keyframe_{position:03d}_frame_{frame.frame_index:08d}.jpg"
        if not cv2.imwrite(str(destination), frame.frame_bgr):
            raise OSError(f"Could not write keyframe image: {destination}")
        keyframe_paths.append(str(destination.resolve()))
    _save_contact_sheet(keyframe_paths, contact_path)
    tensors = {
        "visual_embeddings": selected_features,
        "frame_indices": torch.tensor([frame.frame_index for frame in selected_frames], dtype=torch.int64),
        "timestamps": torch.tensor([frame.timestamp_seconds for frame in selected_frames], dtype=torch.float64),
        "motion_scores": torch.tensor(selected_motion, dtype=torch.float32),
        "keyframe_paths": keyframe_paths,
    }
    cache_path = cache.save(
        video_id, source_hash, cache_parameters, tensors,
        extra_metadata={"source_video_path": str(path)},
    )
    _write_keyframe_metadata(
        metadata_path, video_id, metadata_to_dict(video_metadata), cache_path,
        cache_parameters, selected_frames, selected_motion, keyframe_paths, contact_path,
    )
    return KeyframeSelection(
        video_id, metadata_to_dict(video_metadata), selected_frames, selected_features,
        selected_motion, keyframe_paths, str(cache_path.resolve()), str(metadata_path.resolve()),
        str(contact_path.resolve()), False,
    )

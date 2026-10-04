"""Lightweight frame-difference motion filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from src.data.video_io import SampledFrame


@dataclass
class MotionFilteredFrames:
    frames: list[SampledFrame]
    scores: list[float]
    selected_original_positions: list[int]


def motion_scores(frames: Sequence[SampledFrame]) -> list[float]:
    """Calculate mean absolute grayscale frame difference; the first score is zero."""
    if not frames:
        return []
    grays = [cv2.cvtColor(frame.frame_bgr, cv2.COLOR_BGR2GRAY) for frame in frames]
    scores = [0.0]
    for previous, current in zip(grays, grays[1:]):
        if previous.shape != current.shape:
            current = cv2.resize(current, (previous.shape[1], previous.shape[0]))
        scores.append(float(cv2.absdiff(previous, current).mean() / 255.0))
    return scores


def filter_by_motion(
    frames: Sequence[SampledFrame], threshold: float = 0.015,
) -> MotionFilteredFrames:
    """Retain motion events and both valid sequence endpoints.

    The first and last frames are always retained. If the threshold leaves no interior
    frame, retain the frame with greatest motion so static/low-motion videos stay useful.
    """
    if threshold < 0:
        raise ValueError("motion threshold must be non-negative")
    if not frames:
        return MotionFilteredFrames([], [], [])
    scores = motion_scores(frames)
    positions = {0, len(frames) - 1}
    positions.update(i for i, score in enumerate(scores[1:], start=1) if score >= threshold)
    if len(frames) > 2 and len(positions) == 2:
        positions.add(max(range(1, len(frames) - 1), key=lambda i: scores[i]))
    ordered = sorted(positions)
    return MotionFilteredFrames(
        frames=[frames[i] for i in ordered],
        scores=[scores[i] for i in ordered],
        selected_original_positions=ordered,
    )

"""Safe OpenCV video metadata and frame sampling utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    fps: float
    total_frames: int
    duration_seconds: float
    width: int
    height: int
    codec: str


@dataclass
class SampledFrame:
    frame_index: int
    timestamp_seconds: float
    frame_bgr: np.ndarray


def _fourcc_string(value: float) -> str:
    code = int(value)
    chars = [chr((code >> (8 * index)) & 0xFF) for index in range(4)]
    codec = "".join(chars).strip("\x00 ")
    return codec if codec and codec.isprintable() else "unknown"


def get_video_metadata(video_path: str | Path) -> VideoMetadata:
    """Read basic metadata, raising a clear error if OpenCV cannot open the file."""
    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {path}")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"OpenCV could not open video (unsupported codec or corrupt file): {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"Video has invalid FPS or dimensions: {path}")
        duration = count / fps if count > 0 else 0.0
        return VideoMetadata(
            path=str(path), fps=fps, total_frames=count, duration_seconds=duration,
            width=width, height=height, codec=_fourcc_string(capture.get(cv2.CAP_PROP_FOURCC)),
        )
    finally:
        capture.release()


def iter_sampled_frames(
    video_path: str | Path,
    sample_fps: float = 2.0,
    max_frames: int | None = None,
) -> Iterator[SampledFrame]:
    """Yield decoded frames at approximately ``sample_fps`` while keeping endpoints.

    A sequential decode avoids unreliable frame seeking on variable-frame-rate files.
    The final decoded frame is included when it was not already sampled.
    """
    if sample_fps <= 0:
        raise ValueError("sample_fps must be greater than zero")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be at least 1 when provided")
    metadata = get_video_metadata(video_path)
    capture = cv2.VideoCapture(metadata.path)
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"OpenCV could not reopen video: {metadata.path}")
    step = max(1, int(round(metadata.fps / sample_fps)))
    selected: list[SampledFrame] = []
    last_frame: SampledFrame | None = None
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            current = SampledFrame(index, index / metadata.fps, frame)
            last_frame = current
            if index % step == 0:
                selected.append(current)
            index += 1
        if not selected and last_frame is not None:
            selected.append(last_frame)
        elif last_frame is not None and selected[-1].frame_index != last_frame.frame_index:
            selected.append(last_frame)
    finally:
        capture.release()
    if not selected:
        raise ValueError(f"Video contains no decodable frames: {metadata.path}")
    if max_frames is not None and len(selected) > max_frames:
        if max_frames == 1:
            selected = [selected[0]]
        else:
            positions = np.linspace(0, len(selected) - 1, num=max_frames, dtype=np.int64)
            selected = [selected[int(position)] for position in positions]
    yield from selected


def metadata_to_dict(metadata: VideoMetadata) -> dict[str, object]:
    """Return JSON-serializable metadata."""
    return asdict(metadata)

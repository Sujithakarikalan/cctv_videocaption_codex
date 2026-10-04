"""Phase 2 tests for video reading, keyframe selection, and ResNet18 feature caching."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.data.motion_filter import filter_by_motion
from src.data.video_io import SampledFrame, get_video_metadata, iter_sampled_frames
from src.features.feature_cache import FeatureCache
from src.features.semantic_keyframes import _select_by_similarity, extract_semantic_keyframes
from src.models.resnet18_encoder import ResNet18Encoder


def make_test_video(path: Path, frame_count: int = 8) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("This OpenCV build has no MP4V encoder")
    for index in range(frame_count):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        cv2.rectangle(frame, (index * 5 % 50, 12), (index * 5 % 50 + 12, 30), (20, 220, 60), -1)
        writer.write(frame)
    writer.release()
    return path


def test_video_metadata_and_sampling_include_last_decoded_frame(tmp_path: Path) -> None:
    path = make_test_video(tmp_path / "clip.mp4")
    metadata = get_video_metadata(path)
    assert metadata.fps == pytest.approx(4.0)
    assert metadata.total_frames == 8
    assert metadata.width == 64 and metadata.height == 48
    sampled = list(iter_sampled_frames(path, sample_fps=1.0))
    assert sampled[0].frame_index == 0
    assert sampled[-1].frame_index == 7
    assert all(frame.frame_bgr.shape == (48, 64, 3) for frame in sampled)


def test_unreadable_videos_fail_safely(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        get_video_metadata(tmp_path / "missing.mp4")
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="could not open"):
        get_video_metadata(corrupt)


def test_motion_filter_keeps_endpoints_and_best_interior() -> None:
    frames = [
        SampledFrame(index, float(index), np.full((16, 16, 3), value, dtype=np.uint8))
        for index, value in enumerate((0, 1, 2, 250, 251))
    ]
    selected = filter_by_motion(frames, threshold=1.1)
    assert selected.frames[0].frame_index == 0
    assert selected.frames[-1].frame_index == 4
    assert 3 in [frame.frame_index for frame in selected.frames]


def test_selection_uses_cosine_threshold_and_preserves_first_last() -> None:
    frames = [SampledFrame(i, float(i), np.zeros((8, 8, 3), dtype=np.uint8)) for i in range(4)]
    embeddings = torch.tensor([[1.0, 0], [0.99, 0.01], [0.0, 1.0], [-1.0, 0.0]])
    chosen = _select_by_similarity(embeddings, frames, [0.0, 0.0, 0.0, 0.0], 4, 0.9, 1.0, 0.0)
    assert chosen == [0, 2, 3]


def test_semantic_extraction_writes_nonempty_keyframes_and_reuses_matching_cache(tmp_path: Path) -> None:
    video = make_test_video(tmp_path / "clip.mp4")
    encoder = ResNet18Encoder(pretrained=False).eval()
    kwargs = dict(
        video_path=video, output_dir=tmp_path / "frames", cache_dir=tmp_path / "features",
        encoder=encoder, device="cpu", sample_fps=2, frame_size=64, max_keyframes=4,
        motion_threshold=0.0, semantic_similarity_threshold=0.0, min_gap_seconds=0.0,
    )
    first = extract_semantic_keyframes(**kwargs)
    second = extract_semantic_keyframes(**kwargs)
    assert not first.cache_hit and second.cache_hit
    assert len(first.selected_frames) >= 1
    assert first.selected_frames[0].frame_index == 0
    assert first.selected_frames[-1].frame_index == 7
    assert first.features.shape == (len(first.selected_frames), 512)
    assert torch.allclose(first.features.norm(dim=1), torch.ones(len(first.selected_frames)), atol=1e-5)
    assert Path(first.contact_sheet_path).is_file()
    assert Path(first.metadata_path).is_file()


def test_cache_requires_matching_resnet18_source_and_parameters(tmp_path: Path) -> None:
    cache = FeatureCache(tmp_path)
    parameters = {"encoder_name": "resnet18", "pretrained": False, "semantic_similarity_threshold": 0.88}
    cache.save("video-a", "sha-a", parameters, {"visual_embeddings": torch.ones(1, 512)})
    matching = cache.load("video-a", "sha-a", parameters)
    assert matching is not None
    metadata = matching["metadata"]
    assert metadata["encoder_name"] == "resnet18"
    assert metadata["feature_dimension"] == 512
    assert metadata["pretrained"] is False
    assert cache.load("video-a", "sha-b", parameters) is None
    assert cache.load("video-a", "sha-a", {**parameters, "pretrained": True}) is None
    assert cache.load("video-a", "sha-a", {**parameters, "semantic_similarity_threshold": 0.9}) is None


def test_old_schema_cache_is_rejected(tmp_path: Path) -> None:
    cache = FeatureCache(tmp_path)
    old_path = cache.path_for("legacy-video")
    torch.save({"schema_version": 1, "metadata": {"model_name": "MobileNetV3-Large"}, "features": {}}, old_path)
    assert cache.load("legacy-video", "old-hash", {"encoder_name": "resnet18"}) is None


def test_cli_runs_with_no_pretrained_smoke_mode(tmp_path: Path) -> None:
    import subprocess
    import sys

    project_root = Path(__file__).resolve().parents[1]
    video = make_test_video(tmp_path / "cli_clip.mp4")
    result = subprocess.run(
        [sys.executable, "scripts/extract_keyframes.py", str(video), "--no-pretrained", "--device", "cpu",
         "--output-dir", str(tmp_path / "cli_frames"), "--cache-dir", str(tmp_path / "cli_cache")],
        cwd=project_root, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "random, untrained ResNet18" in result.stderr
    assert "Feature shape:" in result.stdout and "512)" in result.stdout

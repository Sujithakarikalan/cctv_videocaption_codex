import csv
import json

import pytest

from src.data.caption_dataset import (
    load_caption_rows,
    split_caption_manifest,
    validate_caption_manifest,
)
from src.evaluation.caption_evaluation import evaluate_caption_rows
from src.evaluation.caption_metrics import score_captions


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["video_id", "source_id", "video_path", "split", "reference_captions"],
        )
        writer.writeheader()
        writer.writerows(rows)


def row(video_id, source_id, path, split="", captions=None):
    return {
        "video_id": video_id,
        "source_id": source_id,
        "video_path": str(path),
        "split": split,
        "reference_captions": json.dumps(captions if captions is not None else ["A person walks."]),
    }


def test_validation_reports_missing_video_caption_bad_json_and_duplicate_id(tmp_path):
    manifest = tmp_path / "captions.csv"
    write_manifest(manifest, [
        row("same", "src1", tmp_path / "missing.mp4", captions=[]),
        row("same", "src2", tmp_path / "missing2.mp4", captions=[" "]),
    ])
    report = validate_caption_manifest(manifest)
    assert not report["valid"]
    assert any("does not exist" in error for error in report["errors"])
    assert any("duplicate video_id" in error for error in report["errors"])
    assert any("reference_captions" in error for error in report["errors"])


def test_validation_detects_duplicate_path_invalid_split_and_source_leakage(tmp_path):
    video = tmp_path / "video.mp4"
    video.touch()
    manifest = tmp_path / "captions.csv"
    write_manifest(manifest, [
        row("one", "same-source", video, "train"),
        row("two", "same-source", video, "test"),
        row("three", "other", video, "testing"),
    ])
    report = validate_caption_manifest(manifest, require_splits=True)
    assert not report["valid"]
    assert any("duplicate video file" in error for error in report["errors"])
    assert any("Source leakage" in error for error in report["errors"])
    assert any("invalid split" in error for error in report["errors"])


def test_validation_detects_identical_video_content_at_different_paths(tmp_path):
    first, second = tmp_path / "copy1.mp4", tmp_path / "copy2.mp4"
    first.write_bytes(b"same video bytes")
    second.write_bytes(b"same video bytes")
    manifest = tmp_path / "captions.csv"
    write_manifest(manifest, [row("a", "a", first), row("b", "b", second)])
    report = validate_caption_manifest(manifest)
    assert any("duplicate video content" in error for error in report["errors"])


def test_split_groups_original_sources_and_is_deterministic(tmp_path):
    videos = [tmp_path / f"video{i}.mp4" for i in range(5)]
    for index, video in enumerate(videos):
        video.write_bytes(f"distinct video fixture {index}".encode())
    manifest = tmp_path / "input.csv"
    rows = [
        row("clip_a", "source_a", videos[0]),
        row("clip_b", "source_a", videos[1]),
        row("source_c", "source_c", videos[2]),
        row("source_d", "source_d", videos[3]),
        row("source_e", "source_e", videos[4]),
    ]
    write_manifest(manifest, rows)
    out1, out2 = tmp_path / "out1.csv", tmp_path / "out2.csv"
    split_caption_manifest(manifest, out1, seed=11)
    split_caption_manifest(manifest, out2, seed=11)
    assert out1.read_bytes() == out2.read_bytes()
    report = validate_caption_manifest(out1, require_splits=True)
    assert report["valid"], report["errors"]
    loaded = load_caption_rows(out1)
    split_by_source = {entry["source_id"]: entry["split"] for entry in loaded}
    assert split_by_source["source_a"] == loaded[0]["split"] == loaded[1]["split"]
    assert {entry["split"] for entry in loaded} == {"train", "validation", "test"}


def test_split_requires_three_distinct_sources(tmp_path):
    video = tmp_path / "v.mp4"
    video.touch()
    manifest = tmp_path / "input.csv"
    write_manifest(manifest, [row("one", "one", video)])
    with pytest.raises(ValueError, match="At least three"):
        split_caption_manifest(manifest, tmp_path / "output.csv")


def test_evaluation_uses_human_references_and_saves_metrics_without_model(tmp_path):
    rows = [{
        "video_id": "test1", "source_id": "original1", "video_path": str(tmp_path / "v.mp4"),
        "split": "test", "reference_captions": ["A person walks through a public area."],
    }]
    result = evaluate_caption_rows(rows, lambda _path: "A person walks through a public area.")
    assert result["predictions"][0]["generated_caption"] == rows[0]["reference_captions"][0]
    assert result["predictions"][0]["reference_captions"] == rows[0]["reference_captions"]
    assert result["metrics"]["examples"] == 1
    assert result["metrics"]["corpus_bleu_4"] == pytest.approx(1.0)
    assert result["metrics"]["mean_best_reference_rouge_l_f1"] == pytest.approx(1.0)


def test_evaluation_rejects_non_test_rows_and_empty_data():
    with pytest.raises(ValueError, match="test rows only"):
        evaluate_caption_rows([{"video_id": "train", "split": "train"}], lambda _p: "x")
    with pytest.raises(ValueError, match="No test videos"):
        evaluate_caption_rows([], lambda _p: "x")


def test_caption_metrics_use_best_reference():
    scores = score_captions(
        [["A person walks", "A pedestrian is walking"]],
        ["A pedestrian is walking"],
    )
    assert scores["mean_best_reference_rouge_l_f1"] == pytest.approx(1.0)

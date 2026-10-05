"""Held-out caption generation and scoring against human-written references."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from src.evaluation.caption_metrics import score_captions


def evaluate_caption_rows(
    rows: Sequence[dict[str, Any]],
    caption_fn: Callable[[str], Any],
) -> dict[str, Any]:
    """Caption test rows using an injected function, then score against human references."""
    if not rows:
        raise ValueError("No test videos found in caption manifest")
    predictions = []
    references = []
    hypotheses = []
    for row in rows:
        if row.get("split") != "test":
            raise ValueError(f"Evaluation accepts test rows only; {row.get('video_id')} is not test")
        refs = row.get("reference_captions")
        if not isinstance(refs, list) or not refs or any(not isinstance(r, str) or not r.strip() for r in refs):
            raise ValueError(f"Video {row.get('video_id')} has no valid human reference captions")
        generation = caption_fn(row["video_path"])
        caption = generation.caption if hasattr(generation, "caption") else str(generation)
        hypotheses.append(caption)
        references.append(refs)
        predictions.append({
            "video_id": row["video_id"],
            "source_id": row["source_id"],
            "video_path": row["video_path"],
            "split": "test",
            "reference_captions": refs,
            "generated_caption": caption,
            "inference_seconds": getattr(generation, "inference_seconds", None),
            "device": getattr(generation, "device", None),
        })
    return {"metrics": score_captions(references, hypotheses), "predictions": predictions}

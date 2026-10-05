"""Manifest loading, validation, and leakage-safe splits for caption references."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

REQUIRED_COLUMNS = ("video_id", "source_id", "video_path", "split", "reference_captions")
VALID_SPLITS = {"train", "validation", "test"}


def _read_manifest(path: str | Path) -> tuple[Path, list[dict[str, str]], list[str]]:
    manifest = Path(path).expanduser().resolve()
    errors: list[str] = []
    if not manifest.is_file():
        return manifest, [], [f"Manifest file does not exist: {manifest}"]
    try:
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            missing = [name for name in REQUIRED_COLUMNS if name not in fields]
            if missing:
                return manifest, [], ["Missing required columns: " + ", ".join(missing)]
            return manifest, [dict(row) for row in reader], errors
    except (OSError, csv.Error) as exc:
        return manifest, [], [f"Could not read manifest: {exc}"]


def validate_caption_manifest(
    path: str | Path, *, require_splits: bool = False, check_leakage: bool = True
) -> dict[str, Any]:
    """Validate rows; a blank split is allowed while preparing a dataset."""
    manifest, rows, errors = _read_manifest(path)
    warnings: list[str] = []
    ids: dict[str, int] = {}
    paths: dict[str, int] = {}
    fingerprints: dict[str, tuple[str, int]] = {}
    sources: dict[str, set[str]] = defaultdict(set)
    for line, row in enumerate(rows, start=2):
        prefix = f"Line {line}"
        video_id = (row.get("video_id") or "").strip()
        source_id = (row.get("source_id") or "").strip()
        raw_path = (row.get("video_path") or "").strip()
        split = (row.get("split") or "").strip().lower()
        raw_captions = (row.get("reference_captions") or "").strip()
        if not video_id:
            errors.append(f"{prefix}: video_id is missing")
        elif video_id in ids:
            errors.append(f"{prefix}: duplicate video_id {video_id!r} (first at line {ids[video_id]})")
        else:
            ids[video_id] = line
        if not source_id:
            errors.append(f"{prefix}: source_id is missing")
            source_id = video_id
        if not raw_path:
            errors.append(f"{prefix}: video_path is missing")
        else:
            candidate = Path(raw_path).expanduser()
            if not candidate.is_absolute():
                candidate = manifest.parent / candidate
            resolved = candidate.resolve()
            if not resolved.is_file():
                errors.append(f"{prefix}: video file does not exist: {resolved}")
            else:
                digest = hashlib.sha256()
                with resolved.open("rb") as video_handle:
                    for chunk in iter(lambda: video_handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                fingerprint = digest.hexdigest()
                if fingerprint in fingerprints:
                    errors.append(
                        f"{prefix}: duplicate video content at {resolved} (same SHA-256 as "
                        f"{fingerprints[fingerprint][0]}, first at line {fingerprints[fingerprint][1]})"
                    )
                else:
                    fingerprints[fingerprint] = (str(resolved), line)
            canonical_path = str(resolved).casefold()
            if canonical_path in paths:
                errors.append(
                    f"{prefix}: duplicate video file {raw_path!r} (first at line {paths[canonical_path]}); "
                    "use one row per video"
                )
            else:
                paths[canonical_path] = line
        if not raw_captions:
            errors.append(f"{prefix}: reference_captions is missing")
        else:
            try:
                captions = json.loads(raw_captions)
                if not isinstance(captions, list) or not captions:
                    raise ValueError("must be a non-empty JSON array of strings")
                if any(not isinstance(c, str) or not c.strip() for c in captions):
                    raise ValueError("must contain only non-empty strings")
            except (json.JSONDecodeError, ValueError) as exc:
                errors.append(f"{prefix}: invalid reference_captions JSON: {exc}")
        if not split:
            if require_splits:
                errors.append(f"{prefix}: split is missing")
        elif split not in VALID_SPLITS:
            errors.append(f"{prefix}: invalid split {split!r}; expected train, validation, or test")
        elif source_id:
            sources[source_id].add(split)
    for source_id, source_splits in sources.items():
        if check_leakage and len(source_splits) > 1:
            errors.append(
                f"Source leakage for source_id {source_id!r}: appears in "
                + ", ".join(sorted(source_splits))
            )
    if not rows:
        errors.append("Manifest contains no data rows")
    return {
        "manifest": str(manifest), "rows": len(rows), "valid": not errors,
        "errors": errors, "warnings": warnings,
    }


def _load_valid_rows(path: str | Path) -> tuple[Path, list[dict[str, str]]]:
    report = validate_caption_manifest(path, require_splits=False, check_leakage=False)
    if not report["valid"]:
        raise ValueError("Invalid caption manifest:\n- " + "\n- ".join(report["errors"]))
    manifest, rows, _ = _read_manifest(path)
    return manifest, rows


def split_caption_manifest(
    path: str | Path,
    output_path: str | Path,
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    """Assign each original source group to exactly one split, deterministically."""
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(r <= 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("All split ratios must be positive and sum to 1")
    manifest, rows = _load_valid_rows(path)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["source_id"].strip() or row["video_id"].strip()].append(row)
    keys = list(groups)
    if len(keys) < 3:
        raise ValueError("At least three distinct source_id groups are required for train/validation/test")
    random.Random(seed).shuffle(keys)
    # Largest-remainder allocation avoids losing groups on small datasets.
    exact = [len(keys) * r for r in ratios]
    counts = [int(v) for v in exact]
    for idx in sorted(range(3), key=lambda i: exact[i] - counts[i], reverse=True)[:len(keys) - sum(counts)]:
        counts[idx] += 1
    for empty_idx in [idx for idx, count in enumerate(counts) if count == 0]:
        donor_idx = max(range(3), key=lambda idx: counts[idx])
        counts[donor_idx] -= 1
        counts[empty_idx] += 1
    names = ("train", "validation", "test")
    assigned: dict[str, str] = {}
    cursor = 0
    for name, count in zip(names, counts):
        for key in keys[cursor:cursor + count]:
            assigned[key] = name
        cursor += count
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        for row in rows:
            group = row["source_id"].strip() or row["video_id"].strip()
            writer.writerow({**row, "split": assigned[group]})
    return {name: sum(assigned[group] == name for group in groups) for name in names}


def load_caption_rows(path: str | Path, *, split: str | None = None) -> list[dict[str, Any]]:
    """Return validated rows with resolved video paths and parsed caption lists."""
    manifest, rows = _load_valid_rows(path)
    report = validate_caption_manifest(path, require_splits=True)
    if not report["valid"]:
        raise ValueError("Invalid caption manifest:\n- " + "\n- ".join(report["errors"]))
    result = []
    for row in rows:
        row_split = row["split"].strip().lower()
        if split is not None and row_split != split:
            continue
        video_path = Path(row["video_path"]).expanduser()
        if not video_path.is_absolute():
            video_path = manifest.parent / video_path
        result.append({
            "video_id": row["video_id"].strip(),
            "source_id": row["source_id"].strip(),
            "video_path": str(video_path.resolve()),
            "split": row_split,
            "reference_captions": json.loads(row["reference_captions"]),
        })
    return result

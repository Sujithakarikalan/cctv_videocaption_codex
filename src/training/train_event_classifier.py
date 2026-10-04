"""Optional video-level normal/abnormal classifier training on temporal embeddings."""

from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.temporal_dataset import TemporalRecord, TemporalVideoDataset, load_event_directories, load_temporal_csv
from src.evaluation.classification_metrics import binary_classification_metrics, select_event_threshold
from src.models.event_classifier import EventClassifier
from src.models.slowfast_encoder import SlowFastEncoder
from src.utils.device import get_device

LOGGER = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_records(manifest_path: str | Path | None, directory_root: str | Path | None) -> list[TemporalRecord]:
    if bool(manifest_path) == bool(directory_root):
        raise ValueError("Provide exactly one of manifest_path or directory_root")
    return load_temporal_csv(manifest_path) if manifest_path else load_event_directories(directory_root)


def _run_split(
    loader: DataLoader,
    temporal_encoder: SlowFastEncoder,
    event_classifier: EventClassifier,
    device: torch.device,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    mixed_precision: bool,
    gradient_clip_norm: float,
) -> tuple[float, np.ndarray, np.ndarray, list[str]]:
    is_training = optimizer is not None
    temporal_encoder.train(is_training)
    event_classifier.train(is_training)
    total_loss = 0.0
    total_examples = 0
    all_labels: list[int] = []
    all_probabilities: list[float] = []
    all_video_ids: list[str] = []
    for batch in loader:
        clips = batch["clips"].to(device)  # B,K,C,T,H,W; uint8 from OpenCV
        labels = batch["label"].to(device=device, dtype=torch.long)
        batch_size, clip_count, channels, frames, height, width = clips.shape
        flattened = clips.reshape(batch_size * clip_count, channels, frames, height, width)
        if is_training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=mixed_precision and device.type == "cuda"):
                temporal_features = temporal_encoder(flattened)
                temporal_features = temporal_features.view(batch_size, clip_count, -1).mean(dim=1)
                output = event_classifier(temporal_features)
                loss = criterion(output["logits"], labels)
            if is_training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for group in optimizer.param_groups for parameter in group["params"]],
                    gradient_clip_norm,
                )
                scaler.step(optimizer)
                scaler.update()
        total_loss += float(loss.detach().item()) * batch_size
        total_examples += batch_size
        all_labels.extend(labels.detach().cpu().tolist())
        all_probabilities.extend(output["probabilities"][:, 1].detach().float().cpu().tolist())
        all_video_ids.extend(batch["video_id"])
    return (
        total_loss / max(total_examples, 1),
        np.asarray(all_labels, dtype=np.int64),
        np.asarray(all_probabilities, dtype=np.float64),
        all_video_ids,
    )


def train_event_classifier(
    manifest_path: str | Path | None = None,
    directory_root: str | Path | None = None,
    output_dir: str | Path = "checkpoints/event_classifier",
    epochs: int = 20,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    num_frames: int = 32,
    sampling_rate: int = 2,
    frame_size: int = 224,
    clips_per_video: int = 1,
    alpha: int = 4,
    beta_inv: int = 8,
    pretrained_temporal: bool = True,
    allow_fallback: bool = True,
    fine_tune_temporal: bool = True,
    device_name: str = "auto",
    seed: int = 42,
    workers: int = 0,
    patience: int = 5,
    gradient_clip_norm: float = 1.0,
    mixed_precision: bool = True,
    temporal_encoder: SlowFastEncoder | None = None,
) -> dict[str, Any]:
    """Train and checkpoint the optional binary event head, using video-level splits."""
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size, and patience must be positive")
    _set_seed(seed)
    records = load_records(manifest_path, directory_root)
    labels = {record.label.casefold() for record in records}
    if labels == {"normal", "abnormal"}:
        label_to_id = {"normal": 0, "abnormal": 1}
        records = [TemporalRecord(r.video_id, r.video_path, r.label.casefold(), r.split) for r in records]
    elif len(labels) == 2:
        ordered = sorted(labels)
        label_to_id = {label: index for index, label in enumerate(ordered)}
        records = [TemporalRecord(r.video_id, r.video_path, r.label.casefold(), r.split) for r in records]
    else:
        raise ValueError(f"Binary event training needs exactly two classes, found: {sorted(labels)}")
    train_dataset = TemporalVideoDataset(records, "train", num_frames, sampling_rate, frame_size, clips_per_video, label_to_id)
    val_dataset = TemporalVideoDataset(records, "val", num_frames, sampling_rate, frame_size, clips_per_video, label_to_id)
    train_labels = [train_dataset.label_to_id[record.label] for record in train_dataset.records]
    class_counts = np.bincount(train_labels, minlength=2)
    if (class_counts == 0).any():
        raise ValueError("Training split must contain examples from both event classes")
    class_weights = torch.tensor(
        [len(train_labels) / (2.0 * count) for count in class_counts], dtype=torch.float32,
    )
    sample_weights = torch.tensor([1.0 / class_counts[label] for label in train_labels], dtype=torch.double)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    device = get_device(device_name)
    if temporal_encoder is None:
        temporal_encoder = SlowFastEncoder(
            pretrained=pretrained_temporal, allow_fallback=allow_fallback, alpha=alpha, beta_inv=beta_inv,
        )
    if not fine_tune_temporal:
        for parameter in temporal_encoder.parameters():
            parameter.requires_grad = False
    event_classifier = EventClassifier(temporal_encoder.embedding_dim)
    temporal_encoder.to(device)
    event_classifier.to(device)
    parameters = [parameter for parameter in list(temporal_encoder.parameters()) + list(event_classifier.parameters()) if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision and device.type == "cuda")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_threshold = 0.5
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        train_loss, _, _, _ = _run_split(
            train_loader, temporal_encoder, event_classifier, device, criterion, optimizer,
            scaler, mixed_precision, gradient_clip_norm,
        )
        val_loss, val_labels, val_probabilities, _ = _run_split(
            val_loader, temporal_encoder, event_classifier, device, criterion, None,
            scaler, False, gradient_clip_norm,
        )
        threshold = select_event_threshold(val_labels, val_probabilities)
        metrics = binary_classification_metrics(val_labels, val_probabilities, threshold)
        row = {
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"], **metrics,
        }
        history.append(row)
        scheduler.step()
        state = {
            "epoch": epoch,
            "temporal_encoder_state_dict": temporal_encoder.state_dict(),
            "event_classifier_state_dict": event_classifier.state_dict(),
            "event_classifier_config": {"input_dim": temporal_encoder.embedding_dim},
            "label_to_id": label_to_id,
            "decision_threshold": threshold,
            "temporal_backend": temporal_encoder.backend,
            "temporal_pretrained": pretrained_temporal,
            "metrics": row,
        }
        torch.save(state, destination / "last.pt")
        with (destination / "metrics_history.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        (destination / "metrics_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        LOGGER.info("epoch %d/%d train_loss=%.4f val_loss=%.4f val_f1=%.4f", epoch, epochs, train_loss, val_loss, metrics["f1"])
        if metrics["f1"] > best_f1:
            best_f1, best_threshold, epochs_without_improvement = metrics["f1"], threshold, 0
            torch.save(state, destination / "best_val_f1.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    return {
        "output_dir": str(destination.resolve()),
        "best_checkpoint": str((destination / "best_val_f1.pt").resolve()),
        "last_checkpoint": str((destination / "last.pt").resolve()),
        "best_validation_f1": best_f1,
        "decision_threshold": best_threshold,
        "temporal_backend": temporal_encoder.backend,
        "label_to_id": label_to_id,
        "history": history,
    }

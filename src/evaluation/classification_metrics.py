"""Metrics and operating-threshold selection for binary event classification."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_event_threshold(labels: list[int] | np.ndarray, abnormal_probabilities: list[float] | np.ndarray) -> float:
    """Select the validation threshold with highest abnormal-class F1; default 0.5 if empty."""
    y_true = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(abnormal_probabilities, dtype=np.float64)
    if y_true.size == 0 or y_true.shape != probabilities.shape:
        return 0.5
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in candidates:
        score = f1_score(y_true, probabilities >= threshold, zero_division=0)
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), float(score)
    return best_threshold


def binary_classification_metrics(
    labels: list[int] | np.ndarray,
    abnormal_probabilities: list[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Return accuracy, balanced accuracy, precision/recall/F1, AUC, and 2x2 matrix."""
    y_true = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(abnormal_probabilities, dtype=np.float64)
    if y_true.size == 0:
        raise ValueError("Cannot evaluate an empty label list")
    if y_true.shape != probabilities.shape:
        raise ValueError("labels and abnormal_probabilities must have the same shape")
    predictions = (probabilities >= threshold).astype(np.int64)
    auc = float(roc_auc_score(y_true, probabilities)) if np.unique(y_true).size == 2 else None
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": auc,
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
    }

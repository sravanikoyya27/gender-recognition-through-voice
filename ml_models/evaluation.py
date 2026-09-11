"""Shared classification metrics so every model (MLP and scikit-learn) is scored identically.

Class mapping is fixed project-wide: index 0 = male, index 1 = female
(the MLP's one-hot order and the scikit-learn 0/1 labels agree).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

CLASS_NAMES: tuple[str, str] = ("male", "female")
CLASS_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
POSITIVE_INDEX = CLASS_TO_INDEX["female"]  # probability column used for ROC-AUC


def labels_to_index(labels) -> np.ndarray:
    """'male'/'female' strings (any case) -> 0/1 integer array."""
    return np.asarray([CLASS_TO_INDEX[str(lbl).lower()] for lbl in labels], dtype=np.int64)


def metrics_from_probabilities(y_idx: np.ndarray, probs: np.ndarray, split_name: str) -> dict[str, Any]:
    """Accuracy, precision/recall/F1 (per class, macro, weighted), ROC-AUC, log-loss, confusion matrix.

    ``probs`` has shape (n, 2) with columns in CLASS_NAMES order.
    """
    y_idx = np.asarray(y_idx, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    pred = probs.argmax(axis=1)
    labels = list(range(len(CLASS_NAMES)))
    cm = confusion_matrix(y_idx, pred, labels=labels)

    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": float(precision_score(y_idx, pred, labels=[i], average="macro", zero_division=0)),
            "recall": float(recall_score(y_idx, pred, labels=[i], average="macro", zero_division=0)),
            "f1": float(f1_score(y_idx, pred, labels=[i], average="macro", zero_division=0)),
            "support": int((y_idx == i).sum()),
        }

    eps = 1e-12
    log_loss = float(-np.mean(np.log(probs[np.arange(len(y_idx)), y_idx] + eps)))
    return {
        "split": split_name,
        "n_samples": int(len(y_idx)),
        "accuracy": float(accuracy_score(y_idx, pred)),
        "precision_macro": float(precision_score(y_idx, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_idx, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_idx, pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_idx, pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_idx, pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_idx, pred, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_idx, probs[:, POSITIVE_INDEX])),
        "log_loss": log_loss,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": list(CLASS_NAMES),
            "rows": "true",
            "cols": "predicted",
            "matrix": cm.tolist(),
        },
    }


def log_metrics(metrics: dict[str, Any], logger: logging.Logger, prefix: str = "") -> None:
    logger.info(
        "%s%s: n=%d acc=%.4f prec_macro=%.4f rec_macro=%.4f f1_macro=%.4f auc=%.4f",
        prefix,
        metrics["split"],
        metrics["n_samples"],
        metrics["accuracy"],
        metrics["precision_macro"],
        metrics["recall_macro"],
        metrics["f1_macro"],
        metrics["roc_auc"],
    )
    logger.info(
        "%s%s confusion matrix (rows=true %s):\n%s",
        prefix,
        metrics["split"],
        CLASS_NAMES,
        np.asarray(metrics["confusion_matrix"]["matrix"]),
    )

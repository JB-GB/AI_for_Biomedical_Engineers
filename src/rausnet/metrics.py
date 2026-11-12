"""Evaluation metrics (numpy only) for 4-class ordinal grading."""

from __future__ import annotations

import numpy as np

N_CLASSES = 4


def confusion_matrix(y_true, y_pred, n_classes: int = N_CLASSES) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true, int), np.asarray(y_pred, int)), 1)
    return cm


def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else float("nan")


def per_class_recall(y_true, y_pred, n_classes: int = N_CLASSES) -> np.ndarray:
    """Sensitivity per class; NaN for classes absent from ``y_true``."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    support = cm.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(support > 0, np.diag(cm) / support, np.nan)


def balanced_accuracy(y_true, y_pred, n_classes: int = N_CLASSES) -> float:
    return float(np.nanmean(per_class_recall(y_true, y_pred, n_classes)))


def quadratic_weighted_kappa(y_true, y_pred, n_classes: int = N_CLASSES) -> float:
    """Cohen's kappa with quadratic weights (agreement for ordinal grades)."""
    cm = confusion_matrix(y_true, y_pred, n_classes).astype(float)
    n = cm.sum()
    if n == 0:
        return float("nan")
    i, j = np.indices((n_classes, n_classes))
    weights = (i - j) ** 2 / (n_classes - 1) ** 2
    expected = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / n
    denom = (weights * expected).sum()
    return 1.0 if denom == 0 else float(1 - (weights * cm).sum() / denom)


def roc_auc(y_true_binary, scores) -> float:
    """Area under the ROC curve via the rank-sum (Mann-Whitney) formulation."""
    y = np.asarray(y_true_binary, bool)
    s = np.asarray(scores, float)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    sorted_s = s[order]
    i = 0
    while i < len(s):  # average ranks for ties
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def combined_probabilities(p_gs: np.ndarray, p_pd: np.ndarray) -> np.ndarray:
    """P(combined grade) from independent GS and PD class probabilities.

    combined = 0 if GS = 0, else max(GS, PD). Shapes (N, 4) -> (N, 4).
    """
    out = np.zeros_like(p_gs)
    out[:, 0] = p_gs[:, 0]
    for g in range(1, N_CLASSES):
        for d in range(N_CLASSES):
            out[:, max(g, d)] += p_gs[:, g] * p_pd[:, d]
    return out


def classification_report(y_true, probs: np.ndarray) -> dict:
    """Metrics for one 4-class head. ``probs`` has shape (N, 4)."""
    y_true = np.asarray(y_true, int)
    y_pred = probs.argmax(axis=1)
    return {
        "n": len(y_true),
        "accuracy": accuracy(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "recall_per_class": [None if np.isnan(r) else float(r) for r in per_class_recall(y_true, y_pred)],
        "auc_ge2": roc_auc(y_true >= 2, probs[:, 2:].sum(axis=1)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

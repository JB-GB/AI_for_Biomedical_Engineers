"""Evaluation metrics (numpy only) for 4-class ordinal grading."""

from __future__ import annotations

import itertools

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


def consistent_pd_probabilities(p_gs: np.ndarray, p_pd: np.ndarray) -> np.ndarray:
    """P(PD grade) under the scoring rule that PD > 0 requires synovium (GS > 0).

    The PD head is only trained where GS >= 1, so its output is undefined on GS 0 joints:
    P(PD = d) = P(GS = 0) * [d = 0] + P(GS >= 1) * p_pd[d]. Shapes (N, 4) -> (N, 4).
    """
    out = (1.0 - p_gs[:, :1]) * p_pd
    out[:, 0] += p_gs[:, 0]
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


def _group_index(groups) -> list[np.ndarray]:
    """Row indices for each unique group (patient), in a stable order."""
    groups = np.asarray(groups)
    _, inverse = np.unique(groups, return_inverse=True)
    return [np.flatnonzero(inverse == g) for g in range(inverse.max() + 1)] if len(groups) else []


def bootstrap_ci(y_true, y_pred, groups, metric=quadratic_weighted_kappa, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Patient-level (cluster) bootstrap percentile CI of ``metric``.

    Patients are resampled with replacement and all their images move together,
    so correlated images of one patient do not shrink the interval.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    idx = _group_index(groups)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        rows = np.concatenate([idx[i] for i in rng.integers(len(idx), size=len(idx))])
        stats.append(metric(y_true[rows], y_pred[rows]))
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap(y_true, pred_a, pred_b, groups, metric=quadratic_weighted_kappa, n_boot: int = 2000,
                     alpha: float = 0.05, seed: int = 0) -> dict:
    """Paired patient-level bootstrap of ``metric(b) - metric(a)`` on the same images.

    Returns the observed values and difference with its percentile CI.
    """
    y_true, pred_a, pred_b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    idx = _group_index(groups)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        rows = np.concatenate([idx[i] for i in rng.integers(len(idx), size=len(idx))])
        diffs.append(metric(y_true[rows], pred_b[rows]) - metric(y_true[rows], pred_a[rows]))
    lo, hi = np.nanpercentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    a, b = metric(y_true, pred_a), metric(y_true, pred_b)
    return {"a": a, "b": b, "diff": b - a, "ci_low": float(lo), "ci_high": float(hi),
            "n_boot": n_boot, "alpha": alpha, "n_groups": len(idx)}


def reliability_bins(y_true, probs: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Top-label reliability bins: confidence vs accuracy per equal-width confidence bin."""
    y_true = np.asarray(y_true, int)
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    for lo, hi in itertools.pairwise(edges):
        sel = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if sel.any():
            bins.append({"lo": float(lo), "hi": float(hi), "n": int(sel.sum()),
                         "confidence": float(conf[sel].mean()), "accuracy": float(correct[sel].mean())})
    return bins


def expected_calibration_error(y_true, probs: np.ndarray, n_bins: int = 10) -> float:
    """ECE: sample-weighted mean |accuracy - confidence| over top-label confidence bins."""
    n = len(np.asarray(y_true))
    bins = reliability_bins(y_true, probs, n_bins)
    return float(sum(b["n"] / n * abs(b["accuracy"] - b["confidence"]) for b in bins))


def negative_log_likelihood(y_true, probs: np.ndarray) -> float:
    y_true = np.asarray(y_true, int)
    return float(-np.log(np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1)).mean())


def referral_threshold(y_true, probs: np.ndarray, target_accuracy: float = 0.95) -> dict:
    """Lowest confidence threshold whose accepted cases reach ``target_accuracy`` (T4.5).

    Cases with max probability below the threshold are referred to a specialist. Among
    thresholds meeting the target, the one with the highest coverage (fraction graded
    automatically) is chosen. If none meets it, everything is referred (threshold 1.0+).
    """
    y_true = np.asarray(y_true, int)
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    order = np.argsort(-conf, kind="stable")
    correct = (pred[order] == y_true[order]).astype(float)
    acc_at = np.cumsum(correct) / np.arange(1, len(order) + 1)  # accuracy of the top-n most confident
    sorted_conf = conf[order]
    # a threshold can only cut between distinct confidence values
    cut_ok = np.append(sorted_conf[1:] < sorted_conf[:-1], True)
    ok = np.flatnonzero((acc_at >= target_accuracy) & cut_ok)
    if ok.size == 0:
        return {"threshold": float(np.nextafter(1.0, 2.0)), "target_accuracy": target_accuracy,
                "coverage": 0.0, "accuracy_accepted": None, "n": len(y_true)}
    n_acc = int(ok[-1]) + 1
    return {"threshold": float(sorted_conf[n_acc - 1]), "target_accuracy": target_accuracy,
            "coverage": n_acc / len(y_true), "accuracy_accepted": float(acc_at[n_acc - 1]), "n": len(y_true)}


def referral_outcome(y_true, probs: np.ndarray, threshold: float) -> dict:
    """Coverage and accuracy of accepted vs referred cases at a fixed threshold."""
    y_true = np.asarray(y_true, int)
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    keep = conf >= threshold
    acc = lambda m: float((pred[m] == y_true[m]).mean()) if m.any() else None  # noqa: E731
    return {"threshold": threshold, "coverage": float(keep.mean()), "accuracy_accepted": acc(keep),
            "accuracy_referred": acc(~keep), "accuracy_all": acc(np.ones_like(keep)), "n": len(y_true),
            "n_referred": int((~keep).sum())}


def coverage_curve(y_true, probs: np.ndarray, coverages=None) -> list[dict]:
    """Accuracy of the most confident fraction of cases (risk-coverage curve)."""
    y_true = np.asarray(y_true, int)
    coverages = np.linspace(0.5, 1.0, 11) if coverages is None else coverages
    order = np.argsort(-probs.max(axis=1), kind="stable")
    correct = probs.argmax(axis=1)[order] == y_true[order]
    out = []
    for c in coverages:
        n = max(1, round(c * len(order)))
        out.append({"coverage": float(n / len(order)), "accuracy": float(correct[:n].mean()),
                    "min_confidence": float(probs.max(axis=1)[order][n - 1])})
    return out

"""Binary classification metrics for loop detection (numpy only)."""
from __future__ import annotations

import numpy as np


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    return {"tp": int(np.sum(y_true & y_pred)), "fp": int(np.sum(~y_true & y_pred)),
            "fn": int(np.sum(y_true & ~y_pred)), "tn": int(np.sum(~y_true & ~y_pred))}


def at_threshold(y_true, scores, threshold: float = 0.5) -> dict:
    c = confusion(y_true, np.asarray(scores) >= threshold)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return c | {
        "threshold": threshold,
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "balanced_accuracy": (recall + tnr) / 2,
        "false_positive_rate": 1 - tnr,
    }


def roc_auc(y_true, scores) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (ties averaged)."""
    y = np.asarray(y_true, bool)
    s = np.asarray(scores, float)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    # average ranks over ties
    _, first, counts = np.unique(sorted_s, return_index=True, return_counts=True)
    avg = first + (counts - 1) / 2.0 + 1
    ranks[order] = np.repeat(avg, counts)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def precision_recall_curve(y_true, scores):
    """Precision and recall at every distinct score threshold (descending)."""
    y = np.asarray(y_true, bool)
    s = np.asarray(scores, float)
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    distinct = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(y)[distinct]
    fp = (distinct + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / max(y.sum(), 1)
    return precision, recall, s[distinct]


def average_precision(y_true, scores) -> float:
    """Area under the precision-recall curve (step-wise, as in scikit-learn)."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def best_f1(y_true, scores) -> dict:
    precision, recall, thr = precision_recall_curve(y_true, scores)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0)
    k = int(np.argmax(f1))
    return {"f1": float(f1[k]), "precision": float(precision[k]), "recall": float(recall[k]),
            "threshold": float(thr[k])}

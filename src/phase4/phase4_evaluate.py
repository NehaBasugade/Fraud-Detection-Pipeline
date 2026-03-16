from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def recall_at_precision(y_true: np.ndarray, y_score: np.ndarray, min_precision: float) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    valid = np.where(precision[:-1] >= min_precision)[0]
    if len(valid) == 0:
        return 0.0, 1.0
    best_idx = valid[np.argmax(recall[:-1][valid])]
    return float(recall[best_idx]), float(thresholds[best_idx])


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    roc_auc = float(roc_auc_score(y_true, y_score))
    pr_auc = float(average_precision_score(y_true, y_score))
    r80, t80 = recall_at_precision(y_true, y_score, 0.80)
    r90, t90 = recall_at_precision(y_true, y_score, 0.90)
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "recall_at_precision_0_80": r80,
        "threshold_at_precision_0_80": t80,
        "recall_at_precision_0_90": r90,
        "threshold_at_precision_0_90": t90,
    }


def save_metrics(report_dir: Path, split_name: str, metrics: dict) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{split_name}_metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
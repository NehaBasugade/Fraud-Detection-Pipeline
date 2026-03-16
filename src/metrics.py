from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def recall_at_precision_threshold(y_true, y_scores, min_precision: float) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    valid = precision >= min_precision
    if not np.any(valid):
        return 0.0

    return float(np.max(recall[valid]))


def compute_classification_metrics(y_true, y_scores) -> Dict[str, float]:
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_scores)),
        "pr_auc": float(average_precision_score(y_true, y_scores)),
        "recall_at_precision_80": recall_at_precision_threshold(y_true, y_scores, 0.80),
        "recall_at_precision_90": recall_at_precision_threshold(y_true, y_scores, 0.90),
    }
    return metrics

def best_threshold_for_min_precision(y_true, y_scores, min_precision: float):
    """
    Find the threshold that achieves the highest recall while maintaining
    precision >= min_precision.
    """
    from sklearn.metrics import precision_recall_curve
    import numpy as np

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    precision = precision[:-1]
    recall = recall[:-1]

    valid = np.where(precision >= min_precision)[0]

    if len(valid) == 0:
        return {
            "threshold": None,
            "precision": None,
            "recall": 0.0
        }

    best_idx = valid[np.argmax(recall[valid])]

    return {
        "threshold": float(thresholds[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx])
    }


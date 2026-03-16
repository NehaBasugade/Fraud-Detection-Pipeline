from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]

PHASE5_DATA_DIR = ROOT / "artifacts" / "phase5" / "score_only_features"
REPORT_DIR = ROOT / "reports" / "phase5" / "score_only_hybrid"


def recall_at_precision_threshold(y_true, y_score, min_precision: float):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    valid = np.where(precision[:-1] >= min_precision)[0]
    if len(valid) == 0:
        return {"threshold": None, "precision": None, "recall": 0.0}
    idx = valid[np.argmax(recall[:-1][valid])]
    return {
        "threshold": float(thresholds[idx]),
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
    }


def compute_metrics(y_true, y_score):
    p80 = recall_at_precision_threshold(y_true, y_score, 0.80)
    p90 = recall_at_precision_threshold(y_true, y_score, 0.90)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "recall_at_precision_0_80": float(p80["recall"]),
        "threshold_at_precision_0_80": p80["threshold"],
        "recall_at_precision_0_90": float(p90["recall"]),
        "threshold_at_precision_0_90": p90["threshold"],
    }


def fit_lgbm(X_train, y_train, X_val, y_val):
    params = {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": 42,
    }

    train_ds = lgb.Dataset(X_train, label=y_train)
    val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)

    model = lgb.train(
        params,
        train_set=train_ds,
        valid_sets=[val_ds],
        num_boost_round=2000,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )
    return model


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    X_train = np.load(PHASE5_DATA_DIR / "X_train_control.npy")
    X_val = np.load(PHASE5_DATA_DIR / "X_val_control.npy")
    X_test = np.load(PHASE5_DATA_DIR / "X_test_control.npy")

    X_train_h = np.load(PHASE5_DATA_DIR / "X_train_hybrid.npy")
    X_val_h = np.load(PHASE5_DATA_DIR / "X_val_hybrid.npy")
    X_test_h = np.load(PHASE5_DATA_DIR / "X_test_hybrid.npy")

    y_train = np.load(PHASE5_DATA_DIR / "y_train.npy")
    y_val = np.load(PHASE5_DATA_DIR / "y_val.npy")
    y_test = np.load(PHASE5_DATA_DIR / "y_test.npy")

    control_model = fit_lgbm(X_train, y_train, X_val, y_val)
    control_val_pred = control_model.predict(X_val, num_iteration=control_model.best_iteration)
    control_test_pred = control_model.predict(X_test, num_iteration=control_model.best_iteration)

    hybrid_model = fit_lgbm(X_train_h, y_train, X_val_h, y_val)
    hybrid_val_pred = hybrid_model.predict(X_val_h, num_iteration=hybrid_model.best_iteration)
    hybrid_test_pred = hybrid_model.predict(X_test_h, num_iteration=hybrid_model.best_iteration)

    control_val_metrics = compute_metrics(y_val, control_val_pred)
    control_test_metrics = compute_metrics(y_test, control_test_pred)
    hybrid_val_metrics = compute_metrics(y_val, hybrid_val_pred)
    hybrid_test_metrics = compute_metrics(y_test, hybrid_test_pred)

    write_json(REPORT_DIR / "control_val_metrics.json", control_val_metrics)
    write_json(REPORT_DIR / "control_test_metrics.json", control_test_metrics)
    write_json(REPORT_DIR / "hybrid_val_metrics.json", hybrid_val_metrics)
    write_json(REPORT_DIR / "hybrid_test_metrics.json", hybrid_test_metrics)

    summary = {
        "phase": 5,
        "name": "Minimal score-only hybrid",
        "control": {
            "description": "LightGBM on filtered Phase 4 tabular matrix only"
        },
        "hybrid": {
            "description": "LightGBM on same filtered matrix + 1 extra feature = strict GNN score"
        },
        "notes": [
            "train score is strict expanding-window OOF",
            "earliest train rows without leakage-safe OOF scores were dropped from both control and hybrid training",
            "val score uses train history only",
            "test score uses train+val history only",
            "this is the first defendable hybrid"
        ],
        "train_rows_used": int(X_train.shape[0]),
        "original_train_rows": 413378,
        "hybrid_feature_dim": int(X_train_h.shape[1]),
    }
    write_json(REPORT_DIR / "report.json", summary)

    print("Saved report to:")
    print(REPORT_DIR)
    print("\nControl test metrics:")
    print(json.dumps(control_test_metrics, indent=2))
    print("\nHybrid test metrics:")
    print(json.dumps(hybrid_test_metrics, indent=2))


if __name__ == "__main__":
    main()
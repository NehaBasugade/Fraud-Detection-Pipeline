from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder


DROP_COLS = {"TransactionID", "TransactionDT", "isFraud", "D7", "dist2"}
TARGET_COL = "isFraud"
MISSING_TOKEN = "__MISSING__"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_split_data(processed_dir: Path) -> Dict[str, pd.DataFrame]:
    return {
        "train": pd.read_parquet(processed_dir / "train_df.parquet"),
        "val": pd.read_parquet(processed_dir / "val_df.parquet"),
        "test": pd.read_parquet(processed_dir / "test_df.parquet"),
    }


def get_join_key(df: pd.DataFrame) -> str:
    if "TransactionID" in df.columns:
        return "TransactionID"
    if "transaction_node_id" in df.columns:
        return "transaction_node_id"
    raise ValueError("No supported join key found. Expected TransactionID or transaction_node_id.")


def load_gnn_feature_tables(gnn_feature_dir: Path) -> Dict[str, pd.DataFrame]:
    tables = {}
    for split in ("train", "val", "test"):
        path = gnn_feature_dir / f"{split}_gnn_features.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing GNN feature table: {path}")
        tables[split] = pd.read_parquet(path)
    return tables


def merge_gnn_features(
    split_frames: Dict[str, pd.DataFrame],
    gnn_tables: Dict[str, pd.DataFrame],
) -> Tuple[Dict[str, pd.DataFrame], str, List[str]]:
    merged = {}
    join_key = get_join_key(split_frames["train"])

    feature_cols: List[str] | None = None

    for split, df in split_frames.items():
        gnn_df = gnn_tables[split].copy()

        if join_key not in gnn_df.columns:
            raise ValueError(
                f"{split} GNN features missing join key '{join_key}'. "
                f"Columns: {list(gnn_df.columns)}"
            )

        gnn_feature_cols = [c for c in gnn_df.columns if c != join_key]
        if not gnn_feature_cols:
            raise ValueError(f"{split} GNN feature table has no feature columns.")

        if feature_cols is None:
            feature_cols = gnn_feature_cols
        else:
            if feature_cols != gnn_feature_cols:
                raise ValueError(
                    f"GNN feature column mismatch on split={split}. "
                    f"Expected {feature_cols}, found {gnn_feature_cols}"
                )

        merged_df = df.merge(gnn_df, on=join_key, how="left", validate="one_to_one")

        missing_rate = merged_df[gnn_feature_cols].isna().mean().max()
        if missing_rate > 0:
            raise ValueError(
                f"Split={split} has missing merged GNN features. Max missing rate={missing_rate:.6f}"
            )

        merged[split] = merged_df

    assert feature_cols is not None
    return merged, join_key, feature_cols


def get_tabular_feature_columns(df: pd.DataFrame, extra_exclude: List[str] | None = None) -> List[str]:
    exclude = set(DROP_COLS)
    if extra_exclude:
        exclude.update(extra_exclude)
    return [c for c in df.columns if c not in exclude]


def split_feature_types(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[List[str], List[str]]:
    cat_cols: List[str] = []
    num_cols: List[str] = []
    for col in feature_cols:
        dtype = df[col].dtype
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            cat_cols.append(col)
        else:
            num_cols.append(col)
    return num_cols, cat_cols


def fit_ordinal_encoder(train_df: pd.DataFrame, cat_cols: List[str]) -> OrdinalEncoder | None:
    if not cat_cols:
        return None

    enc = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
        dtype=np.int32,
    )
    enc.fit(train_df[cat_cols].fillna(MISSING_TOKEN).astype(str))
    return enc


def transform_features(
    split_frames: Dict[str, pd.DataFrame],
    feature_cols: List[str],
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    train_df = split_frames["train"]
    num_cols, cat_cols = split_feature_types(train_df, feature_cols)
    enc = fit_ordinal_encoder(train_df, cat_cols)

    transformed: Dict[str, pd.DataFrame] = {}
    final_cols = num_cols + cat_cols

    for split, df in split_frames.items():
        out = pd.DataFrame(index=df.index)

        if num_cols:
            out[num_cols] = df[num_cols]

        if cat_cols:
            assert enc is not None
            out[cat_cols] = enc.transform(df[cat_cols].fillna(MISSING_TOKEN).astype(str))

        transformed[split] = out[final_cols]

    return transformed, cat_cols


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    def recall_at_precision(target_precision: float) -> Tuple[float, float]:
        valid = np.where(precision[:-1] >= target_precision)[0]
        if len(valid) == 0:
            return 0.0, 1.0
        idx = valid[np.argmax(recall[:-1][valid])]
        return float(recall[idx]), float(thresholds[idx])

    r80, t80 = recall_at_precision(0.80)
    r90, t90 = recall_at_precision(0.90)

    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "recall_at_precision_0_80": r80,
        "threshold_at_precision_0_80": t80,
        "recall_at_precision_0_90": r90,
        "threshold_at_precision_0_90": t90,
    }


def save_json(obj: dict, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def write_summary_markdown(
    path: Path,
    *,
    run_name: str,
    feature_cols: List[str],
    gnn_feature_cols: List[str],
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    best_iteration: int | None,
) -> None:
    ensure_dir(path.parent)

    lines = [
        f"# {run_name}",
        "",
        "## Feature set",
        f"- Total model features: {len(feature_cols)}",
        f"- GNN-derived features added: {len(gnn_feature_cols)}",
    ]

    if gnn_feature_cols:
        preview = ", ".join(gnn_feature_cols[:12])
        if len(gnn_feature_cols) > 12:
            preview += ", ..."
        lines.append(f"- GNN feature columns: {preview}")

    if best_iteration is not None:
        lines.extend(["", "## Training", f"- Best iteration: {best_iteration}"])

    lines.extend(
        [
            "",
            "## Validation",
            f"- ROC-AUC: {val_metrics['roc_auc']:.6f}",
            f"- PR-AUC: {val_metrics['pr_auc']:.6f}",
            f"- Recall @ Precision >= 0.80: {val_metrics['recall_at_precision_0_80']:.6f}",
            f"- Threshold @ Precision >= 0.80: {val_metrics['threshold_at_precision_0_80']:.6f}",
            f"- Recall @ Precision >= 0.90: {val_metrics['recall_at_precision_0_90']:.6f}",
            f"- Threshold @ Precision >= 0.90: {val_metrics['threshold_at_precision_0_90']:.6f}",
            "",
            "## Test",
            f"- ROC-AUC: {test_metrics['roc_auc']:.6f}",
            f"- PR-AUC: {test_metrics['pr_auc']:.6f}",
            f"- Recall @ Precision >= 0.80: {test_metrics['recall_at_precision_0_80']:.6f}",
            f"- Threshold @ Precision >= 0.80: {test_metrics['threshold_at_precision_0_80']:.6f}",
            f"- Recall @ Precision >= 0.90: {test_metrics['recall_at_precision_0_90']:.6f}",
            f"- Threshold @ Precision >= 0.90: {test_metrics['threshold_at_precision_0_90']:.6f}",
        ]
    )

    path.write_text("\n".join(lines))
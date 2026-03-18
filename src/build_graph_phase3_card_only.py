# src/build_graph_phase3_card_only.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd



TRAIN_PATH = Path("data/processed/train_df.parquet")
VAL_PATH = Path("data/processed/val_df.parquet")
TEST_PATH = Path("data/processed/test_df.parquet")

ARTIFACT_DIR = Path("artifacts/phase3_card_only")
REPORT_PATH = Path("reports/phase3_card_only_report.json")
SUMMARY_PATH = Path("reports/phase3_card_only_summary.md")

TARGET_COL = "isFraud"
TIME_COL = "TransactionDT"
TX_ID_COL = "TransactionID"

CARD_FIELDS = ["card1", "card2", "card3", "card5"]

# Columns used to define graph structure must be removed from tx features.
GRAPH_STRUCTURE_COLUMNS = CARD_FIELDS

# Helpers


def ensure_dirs() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    # Keep chronological order explicit and deterministic.
    train_df = train_df.sort_values([TIME_COL, TX_ID_COL]).reset_index(drop=True)
    val_df = val_df.sort_values([TIME_COL, TX_ID_COL]).reset_index(drop=True)
    test_df = test_df.sort_values([TIME_COL, TX_ID_COL]).reset_index(drop=True)

    return train_df, val_df, test_df


def normalize_component(series: pd.Series) -> pd.Series:
    """
    Convert components to stable strings while preserving missingness.
    Missing values become empty string for key construction, but are also
    tracked separately so all-missing rows can be rejected.
    """
    s = series.copy()

    # Keep NaNs as NaN until after missingness checks.
    # Convert non-null values to string in a stable way.
    non_null_mask = s.notna()
    s = s.astype("object")
    s.loc[non_null_mask] = s.loc[non_null_mask].astype(str)

    # Normalize obvious junk representations.
    s = s.replace(
        {
            "nan": np.nan,
            "None": np.nan,
            "none": np.nan,
            "NaN": np.nan,
            "": np.nan,
        }
    )
    return s


def build_card_entity_key(df: pd.DataFrame, fields: List[str]) -> pd.Series:
    """
    Build a composite card key from selected card columns.

    Rules:
    - If all components are missing, result is NA.
    - Otherwise join present values with field names to reduce collisions.
    """
    norm_cols = []
    for col in fields:
        if col not in df.columns:
            raise ValueError(f"Missing required card field: {col}")
        norm_cols.append(normalize_component(df[col]).rename(col))

    key_df = pd.concat(norm_cols, axis=1)
    non_missing_count = key_df.notna().sum(axis=1)

    def row_to_key(row: pd.Series) -> str | None:
        parts = []
        for c in fields:
            v = row[c]
            if pd.notna(v):
                parts.append(f"{c}={v}")
        if not parts:
            return None
        return "|".join(parts)

    keys = key_df.apply(row_to_key, axis=1)
    keys = pd.Series(keys, index=df.index, name="card_entity_key")
    keys.loc[non_missing_count == 0] = pd.NA
    return keys


def assign_transaction_node_id(df: pd.DataFrame, split_name: str) -> pd.Series:
    return pd.Series(
        [f"{split_name}_tx_{i}" for i in range(len(df))],
        index=df.index,
        name="transaction_node_id",
    )


def build_split_tables(
    df: pd.DataFrame,
    split_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
    - transaction_nodes: tx node table for the split
    - edges: transaction -> card edge table for valid card keys
    """
    out = df.copy()

    out["transaction_node_id"] = assign_transaction_node_id(out, split_name)
    out["card_entity_key"] = build_card_entity_key(out, CARD_FIELDS)

    # Transaction node table retains raw target/time/id for later split-aware training.
    drop_feature_cols = [c for c in GRAPH_STRUCTURE_COLUMNS if c in out.columns]
    feature_cols = [c for c in out.columns if c not in ["card_entity_key"]]

    tx_nodes = out[feature_cols].copy()

    # Remove graph-structure columns from tx features, but retain metadata columns.
    # We preserve TransactionID / TransactionDT / isFraud because they are useful
    # for training pipeline control and evaluation. Downstream model code can choose
    # whether to feed them as features or metadata.
    tx_feature_columns = [
        c for c in tx_nodes.columns
        if c not in drop_feature_cols
    ]

    tx_nodes = tx_nodes[tx_feature_columns].copy()
    tx_nodes["split"] = split_name
    tx_nodes["chronological_index_in_split"] = np.arange(len(tx_nodes), dtype=np.int64)

    valid_edge_mask = out["card_entity_key"].notna()
    edges = out.loc[valid_edge_mask, ["transaction_node_id", "card_entity_key", TIME_COL, TARGET_COL]].copy()
    edges = edges.rename(
        columns={
            "transaction_node_id": "src_transaction_node_id",
            "card_entity_key": "dst_card_entity_key",
            TIME_COL: "transaction_time",
            TARGET_COL: "transaction_label",
        }
    )
    edges["edge_type"] = "transaction_to_card"
    edges["split"] = split_name
    edges["chronological_edge_index_in_split"] = np.arange(len(edges), dtype=np.int64)

    return tx_nodes, edges


def combine_card_nodes(
    train_edges: pd.DataFrame,
    val_edges: pd.DataFrame,
    test_edges: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build card node table with first-seen split/time and split-level activity.
    """
    all_edges = pd.concat([train_edges, val_edges, test_edges], ignore_index=True)

    if all_edges.empty:
        return pd.DataFrame(
            columns=[
                "card_entity_key",
                "first_seen_time",
                "first_seen_split",
                "train_degree",
                "val_degree",
                "test_degree",
                "global_degree",
            ]
        )

    split_degree = (
        all_edges.groupby(["dst_card_entity_key", "split"])
        .size()
        .unstack(fill_value=0)
        .rename_axis(None, axis=1)
        .reset_index()
        .rename(columns={"dst_card_entity_key": "card_entity_key"})
    )

    for split_col in ["train", "val", "test"]:
        if split_col not in split_degree.columns:
            split_degree[split_col] = 0

    first_seen = (
        all_edges.sort_values(["transaction_time", "src_transaction_node_id"])
        .groupby("dst_card_entity_key", as_index=False)
        .first()[["dst_card_entity_key", "transaction_time", "split"]]
        .rename(
            columns={
                "dst_card_entity_key": "card_entity_key",
                "transaction_time": "first_seen_time",
                "split": "first_seen_split",
            }
        )
    )

    nodes = first_seen.merge(split_degree, on="card_entity_key", how="left")
    nodes = nodes.rename(
        columns={
            "train": "train_degree",
            "val": "val_degree",
            "test": "test_degree",
        }
    )
    nodes["global_degree"] = (
        nodes["train_degree"] + nodes["val_degree"] + nodes["test_degree"]
    )
    nodes = nodes.sort_values(["global_degree", "card_entity_key"], ascending=[False, True]).reset_index(drop=True)
    return nodes


def degree_stats_from_edges(edges: pd.DataFrame) -> Dict[str, float]:
    if edges.empty:
        return {
            "n_entities": 0,
            "n_edges": 0,
            "mean_degree": 0.0,
            "median_degree": 0.0,
            "p90_degree": 0.0,
            "p99_degree": 0.0,
            "max_degree": 0.0,
            "singleton_entity_rate": 0.0,
            "top_1pct_entity_edge_share": 0.0,
        }

    degree = edges.groupby("dst_card_entity_key").size().astype(float)
    n_entities = int(degree.shape[0])
    n_edges = int(degree.sum())
    top_k = max(1, int(np.ceil(0.01 * n_entities)))
    top_share = float(degree.sort_values(ascending=False).head(top_k).sum() / n_edges)

    return {
        "n_entities": n_entities,
        "n_edges": n_edges,
        "mean_degree": float(degree.mean()),
        "median_degree": float(degree.median()),
        "p90_degree": float(degree.quantile(0.90)),
        "p99_degree": float(degree.quantile(0.99)),
        "max_degree": float(degree.max()),
        "singleton_entity_rate": float((degree == 1).mean()),
        "top_1pct_entity_edge_share": top_share,
    }


def fraud_rate_by_train_degree_bucket(
    train_edges: pd.DataFrame,
    val_edges: pd.DataFrame,
    test_edges: pd.DataFrame,
) -> Dict[str, List[Dict[str, float]]]:
    """
    Use TRAIN card degree as the reference degree and inspect fraud rates by degree
    bucket in each split. This is diagnostic only and remains leakage-safe because
    the bucket assignment uses train degree.
    """
    if train_edges.empty:
        return {"train": [], "val": [], "test": []}

    train_degree = (
        train_edges.groupby("dst_card_entity_key")
        .size()
        .rename("train_degree")
        .reset_index()
    )

    def bucketize(df_edges: pd.DataFrame, split_name: str) -> List[Dict[str, float]]:
        if df_edges.empty:
            return []

        tmp = df_edges.merge(train_degree, on="dst_card_entity_key", how="left")
        tmp["train_degree"] = tmp["train_degree"].fillna(0).astype(int)

        def bucket_fn(x: int) -> str:
            if x == 0:
                return "unseen_in_train"
            if x == 1:
                return "1"
            if 2 <= x <= 5:
                return "2_5"
            if 6 <= x <= 20:
                return "6_20"
            if 21 <= x <= 100:
                return "21_100"
            if 101 <= x <= 1000:
                return "101_1000"
            return "1000_plus"

        tmp["train_degree_bucket"] = tmp["train_degree"].map(bucket_fn)

        out = (
            tmp.groupby("train_degree_bucket")
            .agg(
                n_transactions=("src_transaction_node_id", "count"),
                fraud_rate=("transaction_label", "mean"),
            )
            .reset_index()
        )

        bucket_order = {
            "unseen_in_train": 0,
            "1": 1,
            "2_5": 2,
            "6_20": 3,
            "21_100": 4,
            "101_1000": 5,
            "1000_plus": 6,
        }
        out["bucket_order"] = out["train_degree_bucket"].map(bucket_order)
        out = out.sort_values("bucket_order").drop(columns="bucket_order")

        return out.to_dict(orient="records")

    return {
        "train": bucketize(train_edges, "train"),
        "val": bucketize(val_edges, "val"),
        "test": bucketize(test_edges, "test"),
    }


def build_connectivity_summary(
    train_edges: pd.DataFrame,
    val_edges: pd.DataFrame,
    test_edges: pd.DataFrame,
    n_train: int,
    n_val: int,
    n_test: int,
) -> Dict[str, Dict[str, float]]:
    def summarize(split_edges: pd.DataFrame, n_total: int) -> Dict[str, float]:
        connected = int(split_edges["src_transaction_node_id"].nunique()) if not split_edges.empty else 0
        return {
            "transactions_total": int(n_total),
            "transactions_with_card_entity": connected,
            "rate_with_card_entity": float(connected / n_total) if n_total > 0 else 0.0,
        }

    return {
        "train": summarize(train_edges, n_train),
        "val": summarize(val_edges, n_val),
        "test": summarize(test_edges, n_test),
        "global": summarize(
            pd.concat([train_edges, val_edges, test_edges], ignore_index=True),
            n_train + n_val + n_test,
        ),
    }


def save_tables(
    train_tx_nodes: pd.DataFrame,
    val_tx_nodes: pd.DataFrame,
    test_tx_nodes: pd.DataFrame,
    train_edges: pd.DataFrame,
    val_edges: pd.DataFrame,
    test_edges: pd.DataFrame,
    card_nodes: pd.DataFrame,
) -> None:
    train_tx_nodes.to_parquet(ARTIFACT_DIR / "train_transaction_nodes.parquet", index=False)
    val_tx_nodes.to_parquet(ARTIFACT_DIR / "val_transaction_nodes.parquet", index=False)
    test_tx_nodes.to_parquet(ARTIFACT_DIR / "test_transaction_nodes.parquet", index=False)

    train_edges.to_parquet(ARTIFACT_DIR / "train_transaction_to_card_edges.parquet", index=False)
    val_edges.to_parquet(ARTIFACT_DIR / "val_transaction_to_card_edges.parquet", index=False)
    test_edges.to_parquet(ARTIFACT_DIR / "test_transaction_to_card_edges.parquet", index=False)

    card_nodes.to_parquet(ARTIFACT_DIR / "card_nodes.parquet", index=False)


def build_report(
    train_tx_nodes: pd.DataFrame,
    val_tx_nodes: pd.DataFrame,
    test_tx_nodes: pd.DataFrame,
    train_edges: pd.DataFrame,
    val_edges: pd.DataFrame,
    test_edges: pd.DataFrame,
    card_nodes: pd.DataFrame,
) -> Dict:
    report = {
        "phase": 3,
        "name": "Graph Construction - Card Only",
        "graph_version": "phase3_card_only_v1",
        "objective": "Finalize a leakage-safe, production-realistic transaction-centered graph using only the surviving card relation.",
        "schema": {
            "node_types": ["transaction", "card_entity"],
            "edge_types": ["transaction_to_card"],
            "transaction_centered": True,
            "entity_entity_edges": False,
            "transaction_transaction_edges": False,
        },
        "entity_construction": {
            "card_entity_fields": CARD_FIELDS,
            "key_name": "card_entity_key",
        },
        "feature_policy": {
            "graph_structure_columns_removed_from_transaction_features": GRAPH_STRUCTURE_COLUMNS,
            "metadata_retained_in_transaction_tables": [TX_ID_COL, TIME_COL, TARGET_COL, "split", "chronological_index_in_split", "transaction_node_id"],
        },
        "leakage_policy": {
            "train_graph_uses_train_only": True,
            "validation_inference_requires_train_plus_prior_validation_history_only": True,
            "test_inference_requires_train_plus_validation_plus_prior_test_history_only": True,
            "no_future_aware_aggregates": True,
            "no_label_derived_entity_features": True,
            "missing_values_do_not_create_shared_hub_nodes": True,
            "note": "Artifacts are saved chronologically by split so downstream training/inference can enforce prefix-only history.",
        },
        "artifact_paths": {
            "artifact_dir": str(ARTIFACT_DIR),
            "train_transaction_nodes": str(ARTIFACT_DIR / "train_transaction_nodes.parquet"),
            "val_transaction_nodes": str(ARTIFACT_DIR / "val_transaction_nodes.parquet"),
            "test_transaction_nodes": str(ARTIFACT_DIR / "test_transaction_nodes.parquet"),
            "train_edges": str(ARTIFACT_DIR / "train_transaction_to_card_edges.parquet"),
            "val_edges": str(ARTIFACT_DIR / "val_transaction_to_card_edges.parquet"),
            "test_edges": str(ARTIFACT_DIR / "test_transaction_to_card_edges.parquet"),
            "card_nodes": str(ARTIFACT_DIR / "card_nodes.parquet"),
        },
        "counts": {
            "train_transactions": int(len(train_tx_nodes)),
            "val_transactions": int(len(val_tx_nodes)),
            "test_transactions": int(len(test_tx_nodes)),
            "card_nodes_total": int(len(card_nodes)),
            "train_edges": int(len(train_edges)),
            "val_edges": int(len(val_edges)),
            "test_edges": int(len(test_edges)),
            "global_edges": int(len(train_edges) + len(val_edges) + len(test_edges)),
        },
        "connectivity": build_connectivity_summary(
            train_edges, val_edges, test_edges,
            len(train_tx_nodes), len(val_tx_nodes), len(test_tx_nodes)
        ),
        "degree_stats": {
            "train_card_degree": degree_stats_from_edges(train_edges),
            "global_card_degree": degree_stats_from_edges(
                pd.concat([train_edges, val_edges, test_edges], ignore_index=True)
            ),
        },
        "fraud_rate_by_train_degree_bucket": fraud_rate_by_train_degree_bucket(
            train_edges, val_edges, test_edges
        ),
        "decision": {
            "mainline_graph_choice": "card_only",
            "address_status": "dropped_from_mainline",
            "device_status": "unavailable_in_processed_data",
            "reason": (
                "Card relation remained structurally usable. "
                "Address required severe pruning and retained only a tiny fraction of edges in v2. "
                "Device could not be recovered from processed parquet candidate columns. "
                "Card-only is the cleanest leakage-safe graph to carry forward into modeling."
            ),
        },
    }
    return report


def write_report_and_summary(report: Dict) -> None:
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    c = report["counts"]
    conn = report["connectivity"]
    deg = report["degree_stats"]["global_card_degree"]

    summary = f"""# Phase 3 Card-Only Graph Summary

## Decision
The Phase 3 mainline graph has been finalized as a **card-only** heterogeneous graph with:
- node types: `transaction`, `card_entity`
- edge type: `transaction_to_card`

Address was dropped from the mainline graph because prior diagnostics showed it was hub-dominated and v2 pruning retained only a very small fraction of edges.
Device was not available in the processed parquet data and is not blocking progress.

## Leakage policy
This graph design preserves the project's non-negotiable temporal rules:
- train graph uses train only
- validation inference uses train + prior validation history only
- test inference uses train + validation + prior test history only
- no future-aware aggregates
- no label-derived entity features
- missing values do not create shared hub nodes

## Entity construction
Card entity key is constructed from:
- {", ".join(CARD_FIELDS)}

Columns used for graph structure are removed from transaction node features:
- {", ".join(GRAPH_STRUCTURE_COLUMNS)}

## Artifact counts
- Train transactions: {c["train_transactions"]:,}
- Validation transactions: {c["val_transactions"]:,}
- Test transactions: {c["test_transactions"]:,}
- Card nodes: {c["card_nodes_total"]:,}
- Train edges: {c["train_edges"]:,}
- Validation edges: {c["val_edges"]:,}
- Test edges: {c["test_edges"]:,}
- Global edges: {c["global_edges"]:,}

## Connectivity
- Train transactions with card entity: {conn["train"]["transactions_with_card_entity"]:,} ({conn["train"]["rate_with_card_entity"]:.4f})
- Validation transactions with card entity: {conn["val"]["transactions_with_card_entity"]:,} ({conn["val"]["rate_with_card_entity"]:.4f})
- Test transactions with card entity: {conn["test"]["transactions_with_card_entity"]:,} ({conn["test"]["rate_with_card_entity"]:.4f})
- Global transactions with card entity: {conn["global"]["transactions_with_card_entity"]:,} ({conn["global"]["rate_with_card_entity"]:.4f})

## Global card degree diagnostics
- Number of card entities: {deg["n_entities"]:,}
- Number of card edges: {deg["n_edges"]:,}
- Mean degree: {deg["mean_degree"]:.2f}
- Median degree: {deg["median_degree"]:.2f}
- P90 degree: {deg["p90_degree"]:.2f}
- P99 degree: {deg["p99_degree"]:.2f}
- Max degree: {deg["max_degree"]:.2f}
- Singleton entity rate: {deg["singleton_entity_rate"]:.4f}
- Top 1% entity edge share: {deg["top_1pct_entity_edge_share"]:.4f}

## Conclusion
The card relation is the only clearly surviving relation in the current processed dataset.
This is the correct graph to carry into the next modeling phase.
Any future attempt to recover device should be treated as a separate improvement branch, not part of the current mainline graph.
"""
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary)


def main() -> None:
    ensure_dirs()

    train_df, val_df, test_df = load_splits()

    train_tx_nodes, train_edges = build_split_tables(train_df, "train")
    val_tx_nodes, val_edges = build_split_tables(val_df, "val")
    test_tx_nodes, test_edges = build_split_tables(test_df, "test")

    card_nodes = combine_card_nodes(train_edges, val_edges, test_edges)

    save_tables(
        train_tx_nodes=train_tx_nodes,
        val_tx_nodes=val_tx_nodes,
        test_tx_nodes=test_tx_nodes,
        train_edges=train_edges,
        val_edges=val_edges,
        test_edges=test_edges,
        card_nodes=card_nodes,
    )

    report = build_report(
        train_tx_nodes=train_tx_nodes,
        val_tx_nodes=val_tx_nodes,
        test_tx_nodes=test_tx_nodes,
        train_edges=train_edges,
        val_edges=val_edges,
        test_edges=test_edges,
        card_nodes=card_nodes,
    )

    write_report_and_summary(report)

    print("Phase 3 card-only graph artifacts saved successfully.")
    print(f"Artifacts directory: {ARTIFACT_DIR}")
    print(f"Report path: {REPORT_PATH}")
    print(f"Summary path: {SUMMARY_PATH}")
    print()
    print("Mainline graph choice: card_only")
    print(f"Train edges: {len(train_edges):,}")
    print(f"Validation edges: {len(val_edges):,}")
    print(f"Test edges: {len(test_edges):,}")
    print(f"Card nodes: {len(card_nodes):,}")


if __name__ == "__main__":
    main()

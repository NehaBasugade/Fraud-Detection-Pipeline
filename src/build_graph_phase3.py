from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd



# Config


DATA_DIR = Path("data/processed")
ARTIFACT_DIR = Path("artifacts/phase3")
REPORT_DIR = Path("reports")

TRAIN_PATH = DATA_DIR / "train_df.parquet"
VAL_PATH = DATA_DIR / "val_df.parquet"
TEST_PATH = DATA_DIR / "test_df.parquet"

GRAPH_VERSION = "phase3_v1"

# Entity schema for Phase 3 V1
CARD_COLS = ["card1", "card2", "card3", "card5"]
ADDRESS_COLS = ["addr1", "addr2"]
DEVICE_COLS = ["DeviceInfo"]  # conservative first pass

DROP_FROM_TX_FEATURES = {
    "TransactionID",
    "TransactionDT",
    "isFraud",
    "split",
    "transaction_node_id",
    "D7",
    "dist2",
    # raw fields converted into graph structure
    *CARD_COLS,
    *ADDRESS_COLS,
    *DEVICE_COLS,
}



# Helpers


def ensure_dirs() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def stable_hash(text: str, n_chars: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n_chars]


def normalize_scalar(x) -> Optional[str]:
    if pd.isna(x):
        return None
    if isinstance(x, str):
        x = x.strip().lower()
        if x == "":
            return None
        return x
    # keep integers clean when floats hold integer-like values
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return None
        if float(x).is_integer():
            return str(int(x))
        return str(float(x))
    return str(x)


def make_composite_key(row: pd.Series, cols: List[str], prefix: str) -> Optional[str]:
    parts = []
    all_missing = True

    for col in cols:
        val = normalize_scalar(row.get(col))
        if val is None:
            parts.append(f"{col}=<NA>")
        else:
            all_missing = False
            parts.append(f"{col}={val}")

    if all_missing:
        return None

    raw_key = f"{prefix}|" + "|".join(parts)
    return stable_hash(raw_key, n_chars=20)


def build_entity_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["card_entity_key"] = df.apply(
        lambda r: make_composite_key(r, CARD_COLS, "card"), axis=1
    )
    df["address_entity_key"] = df.apply(
        lambda r: make_composite_key(r, ADDRESS_COLS, "address"), axis=1
    )
    df["device_entity_key"] = df.apply(
        lambda r: make_composite_key(r, DEVICE_COLS, "device"), axis=1
    )

    return df


def assign_transaction_node_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["TransactionDT", "TransactionID"], kind="mergesort").reset_index(drop=True)
    df["transaction_node_id"] = np.arange(len(df), dtype=np.int64)
    return df


def build_entity_table(
    df: pd.DataFrame,
    key_col: str,
    entity_type: str,
    start_node_id: int = 0,
) -> pd.DataFrame:
    tmp = df.loc[df[key_col].notna(), [key_col, "TransactionDT", "split"]].copy()
    tmp = tmp.rename(columns={key_col: "entity_key"})

    if tmp.empty:
        return pd.DataFrame(
            columns=[
                "entity_node_id",
                "entity_key",
                "entity_type",
                "first_seen_dt",
                "first_seen_split",
                "total_occurrences_full",
                "is_singleton_full",
            ]
        )

    first_seen = (
        tmp.sort_values(["TransactionDT"], kind="mergesort")
        .groupby("entity_key", as_index=False)
        .first()
        .rename(columns={"TransactionDT": "first_seen_dt", "split": "first_seen_split"})
    )

    counts = (
        tmp.groupby("entity_key", as_index=False)
        .size()
        .rename(columns={"size": "total_occurrences_full"})
    )

    entity_df = first_seen.merge(counts, on="entity_key", how="left")
    entity_df["is_singleton_full"] = entity_df["total_occurrences_full"] == 1
    entity_df["entity_type"] = entity_type
    entity_df = entity_df.sort_values(["first_seen_dt", "entity_key"], kind="mergesort").reset_index(drop=True)
    entity_df["entity_node_id"] = np.arange(start_node_id, start_node_id + len(entity_df), dtype=np.int64)

    return entity_df[
        [
            "entity_node_id",
            "entity_key",
            "entity_type",
            "first_seen_dt",
            "first_seen_split",
            "total_occurrences_full",
            "is_singleton_full",
        ]
    ]


def build_edge_table(
    tx_df: pd.DataFrame,
    entity_df: pd.DataFrame,
    key_col: str,
    edge_type: str,
) -> pd.DataFrame:
    if entity_df.empty:
        return pd.DataFrame(
            columns=[
                "src_node_id",
                "dst_node_id",
                "src_type",
                "dst_type",
                "edge_type",
                "TransactionID",
                "TransactionDT",
                "split",
            ]
        )

    key_to_node = entity_df[["entity_key", "entity_node_id", "entity_type"]].copy()

    edges = tx_df.loc[
        tx_df[key_col].notna(),
        ["transaction_node_id", "TransactionID", "TransactionDT", "split", key_col],
    ].copy()

    edges = edges.merge(
        key_to_node,
        left_on=key_col,
        right_on="entity_key",
        how="left",
        validate="many_to_one",
    )

    edges = edges.rename(
        columns={
            "transaction_node_id": "src_node_id",
            "entity_node_id": "dst_node_id",
            "entity_type": "dst_type",
        }
    )
    edges["src_type"] = "transaction"
    edges["edge_type"] = edge_type

    return edges[
        [
            "src_node_id",
            "dst_node_id",
            "src_type",
            "dst_type",
            "edge_type",
            "TransactionID",
            "TransactionDT",
            "split",
        ]
    ].sort_values(["TransactionDT", "TransactionID"], kind="mergesort").reset_index(drop=True)


def build_transaction_nodes(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        [
            "transaction_node_id",
            "TransactionID",
            "TransactionDT",
            "split",
            "isFraud",
        ]
    ].copy()


def build_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in DROP_FROM_TX_FEATURES]
    feature_cols = ["transaction_node_id"] + feature_cols
    return df[feature_cols].copy()


def summarize_degree_stats(edges: pd.DataFrame, entity_df: pd.DataFrame) -> Dict:
    if entity_df.empty:
        return {
            "n_entities": 0,
            "n_edges": 0,
            "mean_degree": 0.0,
            "median_degree": 0.0,
            "p90_degree": 0.0,
            "p99_degree": 0.0,
            "max_degree": 0,
            "singleton_entities": 0,
            "singleton_entity_rate": 0.0,
            "top_hubs": [],
        }

    deg = (
        edges.groupby("dst_node_id", as_index=False)
        .size()
        .rename(columns={"size": "degree"})
    )

    merged = entity_df.merge(deg, left_on="entity_node_id", right_on="dst_node_id", how="left")
    merged["degree"] = merged["degree"].fillna(0).astype(int)

    top_hubs = (
        merged.sort_values(["degree", "entity_key"], ascending=[False, True])
        .head(20)[["entity_key", "degree"]]
        .to_dict(orient="records")
    )

    singleton_entities = int((merged["degree"] == 1).sum())

    return {
        "n_entities": int(len(entity_df)),
        "n_edges": int(len(edges)),
        "mean_degree": float(merged["degree"].mean()),
        "median_degree": float(merged["degree"].median()),
        "p90_degree": float(merged["degree"].quantile(0.90)),
        "p99_degree": float(merged["degree"].quantile(0.99)),
        "max_degree": int(merged["degree"].max()),
        "singleton_entities": singleton_entities,
        "singleton_entity_rate": float(singleton_entities / len(entity_df)),
        "top_hubs": top_hubs,
    }


def connectivity_stats(tx_df: pd.DataFrame) -> Dict:
    connected_cols = ["card_entity_key", "address_entity_key", "device_entity_key"]
    counts = tx_df[connected_cols].notna().sum(axis=1)

    return {
        "transactions_total": int(len(tx_df)),
        "transactions_with_at_least_1_entity": int((counts >= 1).sum()),
        "transactions_with_at_least_2_entities": int((counts >= 2).sum()),
        "transactions_with_all_3_entities": int((counts == 3).sum()),
        "rate_with_at_least_1_entity": float((counts >= 1).mean()),
        "rate_with_at_least_2_entities": float((counts >= 2).mean()),
        "rate_with_all_3_entities": float((counts == 3).mean()),
    }


def fraud_rate_by_train_degree_bucket(
    tx_df: pd.DataFrame,
    edges: pd.DataFrame,
    entity_df: pd.DataFrame,
    entity_name: str,
) -> List[Dict]:
    """
    Diagnostic only. Uses train split only.
    For each train transaction, attach entity degree (computed on train edges only),
    then summarize fraud rate by degree bucket.
    """
    train_tx = tx_df.loc[tx_df["split"] == "train", ["transaction_node_id", "isFraud"]].copy()
    train_edges = edges.loc[edges["split"] == "train"].copy()

    if train_edges.empty:
        return []

    deg = (
        train_edges.groupby("dst_node_id", as_index=False)
        .size()
        .rename(columns={"size": "train_degree"})
    )

    tx_deg = (
        train_edges[["src_node_id", "dst_node_id"]]
        .merge(deg, on="dst_node_id", how="left")
        .rename(columns={"src_node_id": "transaction_node_id"})
    )

    tx_deg = tx_deg.merge(train_tx, on="transaction_node_id", how="inner")

    def bucket_fn(d: int) -> str:
        if d == 1:
            return "1"
        if d <= 2:
            return "2"
        if d <= 5:
            return "3-5"
        if d <= 10:
            return "6-10"
        if d <= 20:
            return "11-20"
        return "21+"

    tx_deg["degree_bucket"] = tx_deg["train_degree"].map(bucket_fn)

    summary = (
        tx_deg.groupby("degree_bucket", as_index=False)
        .agg(
            transactions=("transaction_node_id", "count"),
            fraud_rate=("isFraud", "mean"),
        )
        .sort_values("degree_bucket")
    )

    out = summary.to_dict(orient="records")
    return [
        {
            "entity_type": entity_name,
            "degree_bucket": row["degree_bucket"],
            "transactions": int(row["transactions"]),
            "fraud_rate": float(row["fraud_rate"]),
        }
        for row in out
    ]


def write_json(path: Path, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_phase3_summary_md(metadata: Dict, diagnostics: Dict) -> str:
    lines = []
    lines.append("# Phase 3 Graph Construction Summary")
    lines.append("")
    lines.append(f"Graph version: `{metadata['graph_version']}`")
    lines.append("")
    lines.append("## Schema")
    lines.append("- Node types: transaction, card_entity, address_entity, device_entity")
    lines.append("- Edge types: transaction_to_card, transaction_to_address, transaction_to_device")
    lines.append("- Missing entity values do not create nodes or edges.")
    lines.append("- Raw fields used to define entity structure were removed from transaction node features.")
    lines.append("")
    lines.append("## Counts")
    lines.append(f"- Transactions: {metadata['counts']['transactions']:,}")
    lines.append(f"- Card entities: {metadata['counts']['card_entities']:,}")
    lines.append(f"- Address entities: {metadata['counts']['address_entities']:,}")
    lines.append(f"- Device entities: {metadata['counts']['device_entities']:,}")
    lines.append(f"- Card edges: {metadata['counts']['transaction_card_edges']:,}")
    lines.append(f"- Address edges: {metadata['counts']['transaction_address_edges']:,}")
    lines.append(f"- Device edges: {metadata['counts']['transaction_device_edges']:,}")
    lines.append("")
    lines.append("## Connectivity")
    conn = diagnostics["connectivity"]
    lines.append(f"- Transactions with >=1 entity link: {conn['transactions_with_at_least_1_entity']:,} ({conn['rate_with_at_least_1_entity']:.2%})")
    lines.append(f"- Transactions with >=2 entity links: {conn['transactions_with_at_least_2_entities']:,} ({conn['rate_with_at_least_2_entities']:.2%})")
    lines.append(f"- Transactions with all 3 entity links: {conn['transactions_with_all_3_entities']:,} ({conn['rate_with_all_3_entities']:.2%})")
    lines.append("")
    lines.append("## Degree diagnostics")
    for entity_name in ["card", "address", "device"]:
        d = diagnostics[f"{entity_name}_degree_stats"]
        lines.append(f"### {entity_name.title()} entities")
        lines.append(f"- Mean degree: {d['mean_degree']:.2f}")
        lines.append(f"- Median degree: {d['median_degree']:.2f}")
        lines.append(f"- 90th percentile degree: {d['p90_degree']:.2f}")
        lines.append(f"- 99th percentile degree: {d['p99_degree']:.2f}")
        lines.append(f"- Max degree: {d['max_degree']}")
        lines.append(f"- Singleton rate: {d['singleton_entity_rate']:.2%}")
        lines.append("")
    return "\n".join(lines)



# Main

def main() -> None:
    ensure_dirs()

    # ----------------------------
    # 1. Load split data
    # ----------------------------
    train_df = pd.read_parquet(TRAIN_PATH).copy()
    val_df = pd.read_parquet(VAL_PATH).copy()
    test_df = pd.read_parquet(TEST_PATH).copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    full_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    full_df = full_df.sort_values(["TransactionDT", "TransactionID"], kind="mergesort").reset_index(drop=True)


    # 2. Build entity keys

    full_df = build_entity_keys(full_df)


    # 3. Assign transaction node ids

    full_df = assign_transaction_node_ids(full_df)


    # 4. Build transaction node table

    tx_nodes = build_transaction_nodes(full_df)


    # 5. Build entity node tables

    card_nodes = build_entity_table(full_df, "card_entity_key", "card_entity", start_node_id=0)
    address_nodes = build_entity_table(full_df, "address_entity_key", "address_entity", start_node_id=0)
    device_nodes = build_entity_table(full_df, "device_entity_key", "device_entity", start_node_id=0)


    # 6. Build edge tables

    edges_tx_card = build_edge_table(
        full_df, card_nodes, "card_entity_key", "transaction_to_card"
    )
    edges_tx_address = build_edge_table(
        full_df, address_nodes, "address_entity_key", "transaction_to_address"
    )
    edges_tx_device = build_edge_table(
        full_df, device_nodes, "device_entity_key", "transaction_to_device"
    )


    # 7. Build transaction feature table

    tx_features = build_transaction_features(full_df)


    # 8. Diagnostics

    diagnostics = {
        "connectivity": connectivity_stats(full_df),
        "card_degree_stats": summarize_degree_stats(edges_tx_card, card_nodes),
        "address_degree_stats": summarize_degree_stats(edges_tx_address, address_nodes),
        "device_degree_stats": summarize_degree_stats(edges_tx_device, device_nodes),
        "fraud_rate_by_train_degree_bucket": {
            "card": fraud_rate_by_train_degree_bucket(full_df, edges_tx_card, card_nodes, "card"),
            "address": fraud_rate_by_train_degree_bucket(full_df, edges_tx_address, address_nodes, "address"),
            "device": fraud_rate_by_train_degree_bucket(full_df, edges_tx_device, device_nodes, "device"),
        },
        "missing_rates": {
            "card_entity_key": float(full_df["card_entity_key"].isna().mean()),
            "address_entity_key": float(full_df["address_entity_key"].isna().mean()),
            "device_entity_key": float(full_df["device_entity_key"].isna().mean()),
        },
    }


    # 9. Metadata

    metadata = {
        "graph_version": GRAPH_VERSION,
        "dataset": "IEEE-CIS Fraud Detection train split parquet artifacts from Phase 1",
        "schema": {
            "node_types": ["transaction", "card_entity", "address_entity", "device_entity"],
            "edge_types": [
                "transaction_to_card",
                "transaction_to_address",
                "transaction_to_device",
            ],
        },
        "entity_key_logic": {
            "card_entity": {
                "source_columns": CARD_COLS,
                "missing_policy": "if all source columns missing, no node and no edge",
            },
            "address_entity": {
                "source_columns": ADDRESS_COLS,
                "missing_policy": "if all source columns missing, no node and no edge",
            },
            "device_entity": {
                "source_columns": DEVICE_COLS,
                "missing_policy": "if all source columns missing, no node and no edge",
            },
        },
        "feature_policy": {
            "dropped_from_transaction_node_features": sorted(DROP_FROM_TX_FEATURES),
            "note": "Fields used to define graph entity structure are removed from transaction node features in this first pass.",
        },
        "counts": {
            "transactions": int(len(tx_nodes)),
            "card_entities": int(len(card_nodes)),
            "address_entities": int(len(address_nodes)),
            "device_entities": int(len(device_nodes)),
            "transaction_card_edges": int(len(edges_tx_card)),
            "transaction_address_edges": int(len(edges_tx_address)),
            "transaction_device_edges": int(len(edges_tx_device)),
            "transaction_feature_columns": int(tx_features.shape[1] - 1),
        },
        "split_counts": {
            "train": int((tx_nodes["split"] == "train").sum()),
            "val": int((tx_nodes["split"] == "val").sum()),
            "test": int((tx_nodes["split"] == "test").sum()),
        },
        "leakage_policy": {
            "training_graph": "train only",
            "validation_inference": "train + prior validation history only",
            "test_inference": "train + validation + prior test history only",
            "label_derived_entity_features": False,
            "future_aware_aggregates_allowed": False,
            "missing_values_create_shared_entity_nodes": False,
        },
    }

    # ----------------------------
    # 10. Save artifacts
    # ----------------------------
    tx_nodes.to_parquet(ARTIFACT_DIR / "nodes_transactions.parquet", index=False)
    card_nodes.to_parquet(ARTIFACT_DIR / "nodes_card_entities.parquet", index=False)
    address_nodes.to_parquet(ARTIFACT_DIR / "nodes_address_entities.parquet", index=False)
    device_nodes.to_parquet(ARTIFACT_DIR / "nodes_device_entities.parquet", index=False)

    edges_tx_card.to_parquet(ARTIFACT_DIR / "edges_transaction_card.parquet", index=False)
    edges_tx_address.to_parquet(ARTIFACT_DIR / "edges_transaction_address.parquet", index=False)
    edges_tx_device.to_parquet(ARTIFACT_DIR / "edges_transaction_device.parquet", index=False)

    tx_features.to_parquet(ARTIFACT_DIR / "transaction_node_features.parquet", index=False)

    write_json(ARTIFACT_DIR / "graph_metadata.json", metadata)
    write_json(ARTIFACT_DIR / "graph_diagnostics.json", diagnostics)

    phase3_report = {
        "phase": 3,
        "name": "Graph Construction",
        "graph_version": GRAPH_VERSION,
        "metadata": metadata,
        "diagnostics": diagnostics,
    }
    write_json(REPORT_DIR / "phase3_graph_report.json", phase3_report)

    summary_md = build_phase3_summary_md(metadata, diagnostics)
    (REPORT_DIR / "phase3_graph_summary.md").write_text(summary_md, encoding="utf-8")

    print("Phase 3 graph artifacts saved successfully.")
    print(f"Artifacts directory: {ARTIFACT_DIR}")
    print(f"Reports directory: {REPORT_DIR}")


if __name__ == "__main__":
    main()

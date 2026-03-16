from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Config
# ============================================================

DATA_DIR = Path("data/processed")
ARTIFACT_DIR = Path("artifacts/phase3_v2")
REPORT_DIR = Path("reports")

TRAIN_PATH = DATA_DIR / "train_df.parquet"
VAL_PATH = DATA_DIR / "val_df.parquet"
TEST_PATH = DATA_DIR / "test_df.parquet"

GRAPH_VERSION = "phase3_v2"

# Core entity fields
CARD_COLS = ["card1", "card2", "card3", "card5"]
ADDRESS_COLS = ["addr1", "addr2"]

# Device candidates in priority order.
# Script will use whichever exist and have non-null values.
DEVICE_CANDIDATE_COLS = [
    "DeviceInfo",
    "DeviceType",
    "id_30",
    "id_31",
]

# Address pruning policy based ONLY on train degrees
ENABLE_ADDRESS_PRUNING = True
ADDRESS_MIN_TRAIN_DEGREE = 2
ADDRESS_MAX_TRAIN_DEGREE = 500

# Optional card pruning if later needed; keep off for now
ENABLE_CARD_PRUNING = False
CARD_MIN_TRAIN_DEGREE = 1
CARD_MAX_TRAIN_DEGREE = 5000

BASE_DROP_FROM_TX_FEATURES = {
    "TransactionID",
    "TransactionDT",
    "isFraud",
    "split",
    "transaction_node_id",
    "D7",
    "dist2",
}


# ============================================================
# Helpers
# ============================================================

def ensure_dirs() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def stable_hash(text: str, n_chars: int = 20) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n_chars]


def normalize_scalar(x) -> Optional[str]:
    if pd.isna(x):
        return None

    if isinstance(x, str):
        x = x.strip().lower()
        if x == "":
            return None
        # normalize repeated spaces
        x = " ".join(x.split())
        return x

    if isinstance(x, (int, np.integer)):
        return str(int(x))

    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return None
        if float(x).is_integer():
            return str(int(x))
        return str(float(x))

    return str(x).strip().lower()


def make_composite_key(row: pd.Series, cols: List[str], prefix: str) -> Optional[str]:
    parts = []
    all_missing = True

    for col in cols:
        val = normalize_scalar(row.get(col))
        if val is None:
            parts.append(f"{col}=<NA>")
        else:
            parts.append(f"{col}={val}")
            all_missing = False

    if all_missing:
        return None

    raw_key = f"{prefix}|" + "|".join(parts)
    return stable_hash(raw_key)


def write_json(path: Path, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ============================================================
# Data loading and auditing
# ============================================================

def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_parquet(TRAIN_PATH).copy()
    val_df = pd.read_parquet(VAL_PATH).copy()
    test_df = pd.read_parquet(TEST_PATH).copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    return train_df, val_df, test_df


def column_audit(
    full_df: pd.DataFrame,
    columns: List[str],
) -> pd.DataFrame:
    rows = []
    n = len(full_df)

    for col in columns:
        exists = col in full_df.columns
        if not exists:
            rows.append(
                {
                    "column": col,
                    "exists": False,
                    "non_null_count": 0,
                    "non_null_rate": 0.0,
                    "n_unique_non_null": 0,
                }
            )
            continue

        s = full_df[col]
        non_null = int(s.notna().sum())
        rows.append(
            {
                "column": col,
                "exists": True,
                "non_null_count": non_null,
                "non_null_rate": float(non_null / n),
                "n_unique_non_null": int(s.dropna().nunique()),
            }
        )

    return pd.DataFrame(rows)


def choose_device_source_columns(audit_df: pd.DataFrame) -> List[str]:
    """
    Conservative rule:
    - Keep candidate columns that exist and have at least some non-null values.
    - Prefer using up to 3 columns to avoid over-fragmentation.
    - Priority is already encoded in DEVICE_CANDIDATE_COLS order.
    """
    usable = audit_df.loc[
        (audit_df["exists"]) & (audit_df["non_null_count"] > 0), "column"
    ].tolist()

    usable_in_priority_order = [c for c in DEVICE_CANDIDATE_COLS if c in usable]

    # Keep up to 3 columns to avoid exploding device identity too aggressively.
    return usable_in_priority_order[:3]


# ============================================================
# Key construction
# ============================================================

def build_entity_keys(
    df: pd.DataFrame,
    device_source_cols: List[str],
) -> pd.DataFrame:
    df = df.copy()

    df["card_entity_key"] = df.apply(
        lambda r: make_composite_key(r, CARD_COLS, "card"),
        axis=1,
    )

    df["address_entity_key_raw"] = df.apply(
        lambda r: make_composite_key(r, ADDRESS_COLS, "address"),
        axis=1,
    )

    if len(device_source_cols) > 0:
        df["device_entity_key"] = df.apply(
            lambda r: make_composite_key(r, device_source_cols, "device"),
            axis=1,
        )
    else:
        df["device_entity_key"] = None

    return df


def assign_transaction_node_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["TransactionDT", "TransactionID"], kind="mergesort").reset_index(drop=True).copy()
    df["transaction_node_id"] = np.arange(len(df), dtype=np.int64)
    return df


# ============================================================
# Entity/node table builders
# ============================================================

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


# ============================================================
# Pruning logic
# ============================================================

def compute_train_degrees_for_key(
    df: pd.DataFrame,
    key_col: str,
) -> pd.DataFrame:
    tmp = df.loc[(df["split"] == "train") & (df[key_col].notna()), [key_col]].copy()

    if tmp.empty:
        return pd.DataFrame(columns=["entity_key", "train_degree"])

    deg = (
        tmp.groupby(key_col, as_index=False)
        .size()
        .rename(columns={key_col: "entity_key", "size": "train_degree"})
    )

    return deg


def prune_entity_keys_by_train_degree(
    df: pd.DataFrame,
    key_col: str,
    min_degree: int,
    max_degree: int,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Keep keys whose TRAIN degree falls within [min_degree, max_degree].
    Keys first seen only in val/test will not have train degree and will be dropped
    under this conservative policy.
    """
    df = df.copy()
    original_non_null = int(df[key_col].notna().sum())

    train_deg = compute_train_degrees_for_key(df, key_col)
    keep_keys = set(
        train_deg.loc[
            (train_deg["train_degree"] >= min_degree)
            & (train_deg["train_degree"] <= max_degree),
            "entity_key",
        ].tolist()
    )

    df[f"{key_col}_pruned"] = df[key_col].where(df[key_col].isin(keep_keys), other=None)

    pruned_non_null = int(df[f"{key_col}_pruned"].notna().sum())

    info = {
        "key_col": key_col,
        "original_non_null_edges": original_non_null,
        "retained_non_null_edges": pruned_non_null,
        "dropped_non_null_edges": original_non_null - pruned_non_null,
        "retained_edge_rate": float(pruned_non_null / original_non_null) if original_non_null > 0 else 0.0,
        "min_train_degree": int(min_degree),
        "max_train_degree": int(max_degree),
        "n_train_keys_total": int(len(train_deg)),
        "n_train_keys_retained": int(len(keep_keys)),
    }
    return df, info


# ============================================================
# Feature table
# ============================================================

def build_transaction_features(
    df: pd.DataFrame,
    drop_cols: set,
) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in drop_cols]
    feature_cols = ["transaction_node_id"] + feature_cols
    return df[feature_cols].copy()


# ============================================================
# Diagnostics
# ============================================================

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
        "max_degree": int(merged["degree"].max()) if len(merged) > 0 else 0,
        "singleton_entities": singleton_entities,
        "singleton_entity_rate": float(singleton_entities / len(entity_df)) if len(entity_df) > 0 else 0.0,
        "top_hubs": top_hubs,
    }


def edge_concentration_stats(
    edges: pd.DataFrame,
    entity_df: pd.DataFrame,
    top_fracs: List[float] = [0.01, 0.05],
) -> Dict:
    if entity_df.empty or edges.empty:
        return {"top_entity_edge_share": {}}

    deg = (
        edges.groupby("dst_node_id", as_index=False)
        .size()
        .rename(columns={"size": "degree"})
    ).sort_values("degree", ascending=False).reset_index(drop=True)

    n_entities = len(entity_df)
    result = {}

    for frac in top_fracs:
        k = max(1, int(np.ceil(frac * n_entities)))
        share = deg.head(k)["degree"].sum() / len(edges)
        result[f"top_{int(frac * 100)}pct_entities_share_of_edges"] = float(share)

    return {"top_entity_edge_share": result}


def connectivity_stats(
    tx_df: pd.DataFrame,
    relation_cols: List[str],
) -> Dict:
    counts = tx_df[relation_cols].notna().sum(axis=1)

    return {
        "transactions_total": int(len(tx_df)),
        "transactions_with_at_least_1_entity": int((counts >= 1).sum()),
        "transactions_with_at_least_2_entities": int((counts >= 2).sum()),
        "transactions_with_all_entities": int((counts == len(relation_cols)).sum()),
        "rate_with_at_least_1_entity": float((counts >= 1).mean()),
        "rate_with_at_least_2_entities": float((counts >= 2).mean()),
        "rate_with_all_entities": float((counts == len(relation_cols)).mean()),
    }


def fraud_rate_by_train_degree_bucket(
    tx_df: pd.DataFrame,
    edges: pd.DataFrame,
    entity_name: str,
) -> List[Dict]:
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
        if d == 2:
            return "2"
        if 3 <= d <= 5:
            return "3-5"
        if 6 <= d <= 10:
            return "6-10"
        if 11 <= d <= 20:
            return "11-20"
        if 21 <= d <= 100:
            return "21-100"
        if 101 <= d <= 500:
            return "101-500"
        return "501+"

    tx_deg["degree_bucket"] = tx_deg["train_degree"].map(bucket_fn)

    ordered_buckets = ["1", "2", "3-5", "6-10", "11-20", "21-100", "101-500", "501+"]

    summary = (
        tx_deg.groupby("degree_bucket", as_index=False)
        .agg(
            transactions=("transaction_node_id", "count"),
            fraud_rate=("isFraud", "mean"),
        )
    )

    summary["degree_bucket"] = pd.Categorical(summary["degree_bucket"], categories=ordered_buckets, ordered=True)
    summary = summary.sort_values("degree_bucket")

    out = []
    for row in summary.to_dict(orient="records"):
        out.append(
            {
                "entity_type": entity_name,
                "degree_bucket": str(row["degree_bucket"]),
                "transactions": int(row["transactions"]),
                "fraud_rate": float(row["fraud_rate"]),
            }
        )
    return out


def retention_by_split(
    original_key: pd.Series,
    pruned_key: pd.Series,
    split: pd.Series,
    name: str,
) -> Dict:
    out = {}
    for sp in ["train", "val", "test"]:
        mask = split == sp
        orig = int(original_key[mask].notna().sum())
        kept = int(pruned_key[mask].notna().sum())
        out[sp] = {
            "original_edges": orig,
            "retained_edges": kept,
            "retained_rate": float(kept / orig) if orig > 0 else 0.0,
        }
    return {name: out}


# ============================================================
# Summary text
# ============================================================

def build_phase3_summary_md(metadata: Dict, diagnostics: Dict) -> str:
    lines = []
    lines.append("# Phase 3 v2 Graph Construction Summary")
    lines.append("")
    lines.append(f"Graph version: `{metadata['graph_version']}`")
    lines.append("")
    lines.append("## Schema")
    lines.append(f"- Node types: {', '.join(metadata['schema']['node_types'])}")
    lines.append(f"- Edge types: {', '.join(metadata['schema']['edge_types'])}")
    lines.append("- Missing entity values do not create nodes or edges.")
    lines.append("- Address pruning is based on train-only degree thresholds.")
    lines.append("")
    lines.append("## Counts")
    counts = metadata["counts"]
    lines.append(f"- Transactions: {counts['transactions']:,}")
    lines.append(f"- Card entities: {counts['card_entities']:,}")
    lines.append(f"- Address entities: {counts['address_entities']:,}")
    lines.append(f"- Device entities: {counts['device_entities']:,}")
    lines.append(f"- Card edges: {counts['transaction_card_edges']:,}")
    lines.append(f"- Address edges: {counts['transaction_address_edges']:,}")
    lines.append(f"- Device edges: {counts['transaction_device_edges']:,}")
    lines.append("")
    lines.append("## Device source columns used")
    device_cols = metadata["entity_key_logic"]["device_entity"]["source_columns"]
    lines.append(f"- {device_cols if len(device_cols) > 0 else 'No usable device columns found'}")
    lines.append("")
    lines.append("## Connectivity")
    conn = diagnostics["connectivity"]
    lines.append(f"- Transactions with >=1 entity link: {conn['transactions_with_at_least_1_entity']:,} ({conn['rate_with_at_least_1_entity']:.2%})")
    lines.append(f"- Transactions with >=2 entity links: {conn['transactions_with_at_least_2_entities']:,} ({conn['rate_with_at_least_2_entities']:.2%})")
    lines.append(f"- Transactions with all entity links: {conn['transactions_with_all_entities']:,} ({conn['rate_with_all_entities']:.2%})")
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


# ============================================================
# Main
# ============================================================

def main() -> None:
    ensure_dirs()

    # ----------------------------
    # 1. Load data
    # ----------------------------
    train_df, val_df, test_df = load_splits()
    full_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    full_df = full_df.sort_values(["TransactionDT", "TransactionID"], kind="mergesort").reset_index(drop=True)

    # ----------------------------
    # 2. Audit relevant columns
    # ----------------------------
    audit_columns = CARD_COLS + ADDRESS_COLS + DEVICE_CANDIDATE_COLS
    audit_df = column_audit(full_df, audit_columns)

    device_source_cols = choose_device_source_columns(audit_df)

    # ----------------------------
    # 3. Build raw entity keys
    # ----------------------------
    full_df = build_entity_keys(full_df, device_source_cols=device_source_cols)

    # ----------------------------
    # 4. Optional pruning
    # ----------------------------
    address_pruning_info = {
        "enabled": ENABLE_ADDRESS_PRUNING,
        "applied": False,
    }

    if ENABLE_ADDRESS_PRUNING:
        full_df, prune_info = prune_entity_keys_by_train_degree(
            full_df,
            key_col="address_entity_key_raw",
            min_degree=ADDRESS_MIN_TRAIN_DEGREE,
            max_degree=ADDRESS_MAX_TRAIN_DEGREE,
        )
        full_df["address_entity_key"] = full_df["address_entity_key_raw_pruned"]
        address_pruning_info.update(prune_info)
        address_pruning_info["applied"] = True
    else:
        full_df["address_entity_key"] = full_df["address_entity_key_raw"]

    card_pruning_info = {
        "enabled": ENABLE_CARD_PRUNING,
        "applied": False,
    }

    if ENABLE_CARD_PRUNING:
        full_df, prune_info = prune_entity_keys_by_train_degree(
            full_df,
            key_col="card_entity_key",
            min_degree=CARD_MIN_TRAIN_DEGREE,
            max_degree=CARD_MAX_TRAIN_DEGREE,
        )
        full_df["card_entity_key_final"] = full_df["card_entity_key_pruned"]
        card_pruning_info.update(prune_info)
        card_pruning_info["applied"] = True
    else:
        full_df["card_entity_key_final"] = full_df["card_entity_key"]

    # ----------------------------
    # 5. Transaction node ids
    # ----------------------------
    full_df = assign_transaction_node_ids(full_df)

    # ----------------------------
    # 6. Build node tables
    # ----------------------------
    tx_nodes = build_transaction_nodes(full_df)

    card_nodes = build_entity_table(
        full_df,
        key_col="card_entity_key_final",
        entity_type="card_entity",
        start_node_id=0,
    )
    address_nodes = build_entity_table(
        full_df,
        key_col="address_entity_key",
        entity_type="address_entity",
        start_node_id=0,
    )
    device_nodes = build_entity_table(
        full_df,
        key_col="device_entity_key",
        entity_type="device_entity",
        start_node_id=0,
    )

    # ----------------------------
    # 7. Build edge tables
    # ----------------------------
    edges_tx_card = build_edge_table(
        full_df,
        card_nodes,
        key_col="card_entity_key_final",
        edge_type="transaction_to_card",
    )
    edges_tx_address = build_edge_table(
        full_df,
        address_nodes,
        key_col="address_entity_key",
        edge_type="transaction_to_address",
    )
    edges_tx_device = build_edge_table(
        full_df,
        device_nodes,
        key_col="device_entity_key",
        edge_type="transaction_to_device",
    )

    # ----------------------------
    # 8. Transaction feature table
    # ----------------------------
    drop_from_tx_features = set(BASE_DROP_FROM_TX_FEATURES)
    drop_from_tx_features.update(CARD_COLS)
    drop_from_tx_features.update(ADDRESS_COLS)
    drop_from_tx_features.update(device_source_cols)

    # also remove intermediate key columns
    drop_from_tx_features.update(
        {
            "card_entity_key",
            "card_entity_key_pruned",
            "card_entity_key_final",
            "address_entity_key_raw",
            "address_entity_key_raw_pruned",
            "address_entity_key",
            "device_entity_key",
        }
    )

    tx_features = build_transaction_features(full_df, drop_from_tx_features)

    # ----------------------------
    # 9. Diagnostics
    # ----------------------------
    relation_cols = ["card_entity_key_final", "address_entity_key", "device_entity_key"]

    diagnostics = {
        "column_audit": audit_df.to_dict(orient="records"),
        "connectivity": connectivity_stats(full_df, relation_cols=relation_cols),
        "card_degree_stats": summarize_degree_stats(edges_tx_card, card_nodes),
        "address_degree_stats": summarize_degree_stats(edges_tx_address, address_nodes),
        "device_degree_stats": summarize_degree_stats(edges_tx_device, device_nodes),
        "card_edge_concentration": edge_concentration_stats(edges_tx_card, card_nodes),
        "address_edge_concentration": edge_concentration_stats(edges_tx_address, address_nodes),
        "device_edge_concentration": edge_concentration_stats(edges_tx_device, device_nodes),
        "fraud_rate_by_train_degree_bucket": {
            "card": fraud_rate_by_train_degree_bucket(full_df, edges_tx_card, "card"),
            "address": fraud_rate_by_train_degree_bucket(full_df, edges_tx_address, "address"),
            "device": fraud_rate_by_train_degree_bucket(full_df, edges_tx_device, "device"),
        },
        "missing_rates": {
            "card_entity_key_final": float(full_df["card_entity_key_final"].isna().mean()),
            "address_entity_key": float(full_df["address_entity_key"].isna().mean()),
            "device_entity_key": float(full_df["device_entity_key"].isna().mean()),
        },
        "pruning": {
            "address": address_pruning_info,
            "card": card_pruning_info,
            "address_retention_by_split": retention_by_split(
                full_df["address_entity_key_raw"],
                full_df["address_entity_key"],
                full_df["split"],
                "address",
            )["address"],
        },
    }

    # ----------------------------
    # 10. Metadata
    # ----------------------------
    metadata = {
        "graph_version": GRAPH_VERSION,
        "dataset": "IEEE-CIS Fraud Detection processed parquet artifacts from Phase 1",
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
                "pruning_policy": {
                    "enabled": ENABLE_CARD_PRUNING,
                    "train_degree_range": [CARD_MIN_TRAIN_DEGREE, CARD_MAX_TRAIN_DEGREE],
                },
            },
            "address_entity": {
                "source_columns": ADDRESS_COLS,
                "missing_policy": "if all source columns missing, no node and no edge",
                "pruning_policy": {
                    "enabled": ENABLE_ADDRESS_PRUNING,
                    "train_degree_range": [ADDRESS_MIN_TRAIN_DEGREE, ADDRESS_MAX_TRAIN_DEGREE],
                },
            },
            "device_entity": {
                "source_columns": device_source_cols,
                "candidate_columns_checked": DEVICE_CANDIDATE_COLS,
                "missing_policy": "if all source columns missing, no node and no edge",
            },
        },
        "feature_policy": {
            "dropped_from_transaction_node_features": sorted(drop_from_tx_features),
            "note": "Fields used to define graph entity structure are removed from transaction node features in v2.",
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
            "address_pruning_thresholds_selected_from": "train degrees only",
        },
    }

    # ----------------------------
    # 11. Save artifacts
    # ----------------------------
    tx_nodes.to_parquet(ARTIFACT_DIR / "nodes_transactions.parquet", index=False)
    card_nodes.to_parquet(ARTIFACT_DIR / "nodes_card_entities.parquet", index=False)
    address_nodes.to_parquet(ARTIFACT_DIR / "nodes_address_entities.parquet", index=False)
    device_nodes.to_parquet(ARTIFACT_DIR / "nodes_device_entities.parquet", index=False)

    edges_tx_card.to_parquet(ARTIFACT_DIR / "edges_transaction_card.parquet", index=False)
    edges_tx_address.to_parquet(ARTIFACT_DIR / "edges_transaction_address.parquet", index=False)
    edges_tx_device.to_parquet(ARTIFACT_DIR / "edges_transaction_device.parquet", index=False)

    tx_features.to_parquet(ARTIFACT_DIR / "transaction_node_features.parquet", index=False)

    audit_df.to_parquet(ARTIFACT_DIR / "column_audit.parquet", index=False)

    write_json(ARTIFACT_DIR / "graph_metadata.json", metadata)
    write_json(ARTIFACT_DIR / "graph_diagnostics.json", diagnostics)

    phase3_report = {
        "phase": 3,
        "name": "Graph Construction",
        "graph_version": GRAPH_VERSION,
        "metadata": metadata,
        "diagnostics": diagnostics,
    }
    write_json(REPORT_DIR / "phase3_graph_report_v2.json", phase3_report)

    summary_md = build_phase3_summary_md(metadata, diagnostics)
    (REPORT_DIR / "phase3_graph_summary_v2.md").write_text(summary_md, encoding="utf-8")

    print("Phase 3 v2 graph artifacts saved successfully.")
    print(f"Artifacts directory: {ARTIFACT_DIR}")
    print(f"Reports directory: {REPORT_DIR}")
    print("\nDevice source columns used:", device_source_cols if device_source_cols else "None")
    print("Address pruning enabled:", ENABLE_ADDRESS_PRUNING)
    if ENABLE_ADDRESS_PRUNING:
        print(
            f"Address train degree range kept: [{ADDRESS_MIN_TRAIN_DEGREE}, {ADDRESS_MAX_TRAIN_DEGREE}]"
        )
        print("Address retained edge rate:", round(address_pruning_info.get("retained_edge_rate", 0.0), 4))


if __name__ == "__main__":
    main()

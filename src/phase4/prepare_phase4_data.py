from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)
from sklearn.preprocessing import OrdinalEncoder


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
PHASE3_DIR = ROOT / "artifacts" / "phase3" / "card_only_mainline"
OUT_DIR = ROOT / "artifacts" / "phase4" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DROP_COLS = {"TransactionID", "TransactionDT", "isFraud", "D7", "dist2"}
MISSING_CAT_TOKEN = "__MISSING__"


def load_split_df(split: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f"{split}_df.parquet")


def load_card_edges(split: str) -> pd.DataFrame:
    return pd.read_parquet(PHASE3_DIR / f"{split}_transaction_to_card_edges.parquet")


def load_txn_nodes(split: str) -> pd.DataFrame:
    return pd.read_parquet(PHASE3_DIR / f"{split}_transaction_nodes.parquet")


def find_first_existing(df: pd.DataFrame, candidates: list[str], df_name: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"{df_name} missing required columns. Found columns: {list(df.columns)}")


def build_transaction_card_map_from_phase3(split: str) -> pd.DataFrame:
    txn_nodes = load_txn_nodes(split).copy()
    edges = load_card_edges(split).copy()

    txn_node_col = find_first_existing(
        txn_nodes,
        ["transaction_node_id", "txn_node_id", "node_id", "transaction_id"],
        f"{split}_transaction_nodes",
    )
    txn_id_col = find_first_existing(
        txn_nodes,
        ["TransactionID", "transaction_id_raw", "raw_TransactionID"],
        f"{split}_transaction_nodes",
    )

    edge_txn_col = find_first_existing(
        edges,
        ["transaction_node_id", "txn_node_id", "src_transaction_node_id", "src", "source", "transaction_id"],
        f"{split}_transaction_to_card_edges",
    )
    edge_card_col = find_first_existing(
        edges,
        ["card_node_id", "dst_card_entity_key", "dst", "target", "card_id"],
        f"{split}_transaction_to_card_edges",
    )

    merged = txn_nodes[[txn_node_col, txn_id_col]].merge(
        edges[[edge_txn_col, edge_card_col]],
        left_on=txn_node_col,
        right_on=edge_txn_col,
        how="inner",
    )

    out = merged[[txn_id_col, edge_card_col]].copy()
    out.columns = ["TransactionID", "card_node_id"]

    if out["TransactionID"].duplicated().any():
        dupes = int(out["TransactionID"].duplicated().sum())
        raise ValueError(f"Duplicate TransactionID values after Phase 3 merge for split={split}. duplicates={dupes}")

    return out


def build_transaction_card_map_fallback(split_df: pd.DataFrame) -> pd.DataFrame:
    card_cols = [c for c in ["card1", "card2", "card3", "card4", "card5", "card6"] if c in split_df.columns]
    if not card_cols:
        raise ValueError("Fallback failed: no card1-card6 columns found in processed split dataframe.")

    tmp = split_df[["TransactionID"] + card_cols].copy()
    for c in card_cols:
        tmp[c] = tmp[c].astype("string").fillna(MISSING_CAT_TOKEN)

    tmp["card_node_id"] = tmp[card_cols].agg("|".join, axis=1)

    if tmp["TransactionID"].duplicated().any():
        raise ValueError("Processed split dataframe has duplicate TransactionID values.")

    return tmp[["TransactionID", "card_node_id"]]


def build_transaction_card_map(split: str, split_df: pd.DataFrame) -> pd.DataFrame:
    try:
        return build_transaction_card_map_from_phase3(split)
    except Exception as e:
        print(f"[WARN] Phase 3 artifact schema mismatch for split={split}. Falling back to card1-card6 reconstruction.")
        print(f"[WARN] Original error: {e}")
        return build_transaction_card_map_fallback(split_df)


def is_categorical_like(series: pd.Series) -> bool:
    dtype = series.dtype
    return (
        is_object_dtype(dtype)
        or is_string_dtype(dtype)
        or is_bool_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    )


def infer_feature_columns(train_df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    feature_cols = [c for c in train_df.columns if c not in DROP_COLS and c != "card_node_id"]

    cat_cols = []
    num_cols = []

    for c in feature_cols:
        s = train_df[c]
        if is_categorical_like(s):
            cat_cols.append(c)
        elif is_numeric_dtype(s):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    return feature_cols, num_cols, cat_cols


def prepare_categorical_frame(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    if not cat_cols:
        return pd.DataFrame(index=df.index)

    out = df[cat_cols].copy()
    for c in cat_cols:
        out[c] = out[c].astype("string").fillna(MISSING_CAT_TOKEN)
        out[c] = out[c].replace("<NA>", MISSING_CAT_TOKEN)
        out[c] = out[c].astype(str)
    return out


def encode_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if num_cols:
        train_num = train_df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32").to_numpy()
        val_num = val_df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32").to_numpy()
        test_num = test_df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32").to_numpy()
    else:
        train_num = np.empty((len(train_df), 0), dtype=np.float32)
        val_num = np.empty((len(val_df), 0), dtype=np.float32)
        test_num = np.empty((len(test_df), 0), dtype=np.float32)

    if cat_cols:
        train_cat_df = prepare_categorical_frame(train_df, cat_cols)
        val_cat_df = prepare_categorical_frame(val_df, cat_cols)
        test_cat_df = prepare_categorical_frame(test_df, cat_cols)

        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        train_cat = enc.fit_transform(train_cat_df).astype("float32")
        val_cat = enc.transform(val_cat_df).astype("float32")
        test_cat = enc.transform(test_cat_df).astype("float32")
    else:
        train_cat = np.empty((len(train_df), 0), dtype=np.float32)
        val_cat = np.empty((len(val_df), 0), dtype=np.float32)
        test_cat = np.empty((len(test_df), 0), dtype=np.float32)

    x_train = np.concatenate([train_num, train_cat], axis=1).astype("float32")
    x_val = np.concatenate([val_num, val_cat], axis=1).astype("float32")
    x_test = np.concatenate([test_num, test_cat], axis=1).astype("float32")

    return x_train, x_val, x_test


def attach_card_ids(df: pd.DataFrame, split: str) -> pd.DataFrame:
    txn_card = build_transaction_card_map(split, df)
    out = df.merge(txn_card, on="TransactionID", how="inner", validate="one_to_one")
    if len(out) != len(df):
        raise ValueError(f"Card merge dropped rows for split={split}: {len(df)} -> {len(out)}")
    return out


def main() -> None:
    train_raw = load_split_df("train")
    val_raw = load_split_df("val")
    test_raw = load_split_df("test")

    train_df = attach_card_ids(train_raw, "train")
    val_df = attach_card_ids(val_raw, "val")
    test_df = attach_card_ids(test_raw, "test")

    feature_cols, num_cols, cat_cols = infer_feature_columns(train_df)
    x_train, x_val, x_test = encode_features(train_df, val_df, test_df, num_cols, cat_cols)

    all_card_ids = pd.Index(
        pd.concat(
            [train_df["card_node_id"], val_df["card_node_id"], test_df["card_node_id"]],
            axis=0,
            ignore_index=True,
        ).unique()
    )
    card_to_idx = {card_id: i for i, card_id in enumerate(all_card_ids.tolist())}

    train_card_idx = train_df["card_node_id"].map(card_to_idx).astype("int64").to_numpy()
    val_card_idx = val_df["card_node_id"].map(card_to_idx).astype("int64").to_numpy()
    test_card_idx = test_df["card_node_id"].map(card_to_idx).astype("int64").to_numpy()

    y_train = train_df["isFraud"].astype("float32").to_numpy()
    y_val = val_df["isFraud"].astype("float32").to_numpy()
    y_test = test_df["isFraud"].astype("float32").to_numpy()

    t_train = train_df["TransactionDT"].astype("int64").to_numpy()
    t_val = val_df["TransactionDT"].astype("int64").to_numpy()
    t_test = test_df["TransactionDT"].astype("int64").to_numpy()

    np.save(OUT_DIR / "X_train.npy", x_train)
    np.save(OUT_DIR / "X_val.npy", x_val)
    np.save(OUT_DIR / "X_test.npy", x_test)

    np.save(OUT_DIR / "y_train.npy", y_train)
    np.save(OUT_DIR / "y_val.npy", y_val)
    np.save(OUT_DIR / "y_test.npy", y_test)

    np.save(OUT_DIR / "card_idx_train.npy", train_card_idx)
    np.save(OUT_DIR / "card_idx_val.npy", val_card_idx)
    np.save(OUT_DIR / "card_idx_test.npy", test_card_idx)

    np.save(OUT_DIR / "time_train.npy", t_train)
    np.save(OUT_DIR / "time_val.npy", t_val)
    np.save(OUT_DIR / "time_test.npy", t_test)

    meta = {
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "n_cards": int(len(all_card_ids)),
        "n_features": int(x_train.shape[1]),
        "feature_columns": feature_cols,
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
    }
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Phase 4 data prepared.")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
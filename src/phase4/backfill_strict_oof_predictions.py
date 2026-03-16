from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.phase4.phase4_dataset import SplitData, build_card_history_index
from src.phase4.phase4_evaluate import compute_metrics
from src.phase4.phase4_infer import (
    RunningCardStats,
    get_recent_history_for_cards,
    make_chronological_batches,
    predict_gnn_strict,
)
from src.phase4.phase4_models import CardHistorySAGE


ROOT = Path(__file__).resolve().parents[2]

PHASE4_DATA_DIR = ROOT / "artifacts" / "phase4" / "data"
PHASE4_MODEL_DIR = ROOT / "artifacts" / "phase4" / "models" / "baseline_gnn_card_only"
PHASE4_PRED_DIR = ROOT / "artifacts" / "phase4" / "predictions" / "baseline_gnn_card_only"
PROCESSED_DIR = ROOT / "data" / "processed"


def load_arrays() -> dict[str, np.ndarray | dict]:
    out = {
        "X_train": np.load(PHASE4_DATA_DIR / "X_train.npy"),
        "X_val": np.load(PHASE4_DATA_DIR / "X_val.npy"),
        "X_test": np.load(PHASE4_DATA_DIR / "X_test.npy"),
        "y_train": np.load(PHASE4_DATA_DIR / "y_train.npy"),
        "y_val": np.load(PHASE4_DATA_DIR / "y_val.npy"),
        "y_test": np.load(PHASE4_DATA_DIR / "y_test.npy"),
        "card_idx_train": np.load(PHASE4_DATA_DIR / "card_idx_train.npy"),
        "card_idx_val": np.load(PHASE4_DATA_DIR / "card_idx_val.npy"),
        "card_idx_test": np.load(PHASE4_DATA_DIR / "card_idx_test.npy"),
        "time_train": np.load(PHASE4_DATA_DIR / "time_train.npy"),
        "time_val": np.load(PHASE4_DATA_DIR / "time_val.npy"),
        "time_test": np.load(PHASE4_DATA_DIR / "time_test.npy"),
    }
    out["metadata"] = json.loads((PHASE4_DATA_DIR / "metadata.json").read_text())
    return out


def make_split(X, y, card_idx, time) -> SplitData:
    return SplitData(
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype=np.float32),
        card_idx=np.asarray(card_idx, dtype=np.int64),
        time=np.asarray(time),
    )


def amt_column_index(feature_columns: list[str]) -> int:
    for candidate in ["TransactionAmt", "transactionamt", "transaction_amt"]:
        if candidate in feature_columns:
            return feature_columns.index(candidate)
    raise ValueError("Could not find TransactionAmt column in feature_columns")


def make_blocks(n: int, n_blocks: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, n, n_blocks + 1, dtype=int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_blocks)]


def fit_gnn_on_history(
    train_split: SplitData,
    val_split: SplitData,
    *,
    n_cards: int,
    feature_columns: list[str],
    device: torch.device,
    hidden_dim: int = 128,
    dropout: float = 0.2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 4,
    batch_size: int = 2048,
    max_hist_per_card: int = 50,
) -> CardHistorySAGE:
    amt_idx = amt_column_index(feature_columns)

    model = CardHistorySAGE(
        txn_in_dim=train_split.X.shape[1],
        card_feat_dim=5,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    pos_weight = (len(train_split.y) - train_split.y.sum()) / max(train_split.y.sum(), 1.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
    )
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    card_hist_idx = build_card_history_index(train_split.card_idx)

    best_val_pr = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        stats = RunningCardStats(n_cards=n_cards)

        for batch in make_chronological_batches(train_split.n, batch_size):
            target_idx = batch.target_idx
            history_idx = batch.history_idx

            target_x_np = train_split.X[target_idx]
            target_y_np = train_split.y[target_idx]
            target_cards_np = train_split.card_idx[target_idx]

            hist_idx = get_recent_history_for_cards(
                target_cards=target_cards_np,
                history_pool_idx=history_idx,
                card_history_index=card_hist_idx,
                max_hist_per_card=max_hist_per_card,
            )

            hist_x_np = (
                train_split.X[hist_idx]
                if hist_idx.size > 0
                else np.empty((0, train_split.X.shape[1]), dtype=np.float32)
            )
            hist_cards_np = (
                train_split.card_idx[hist_idx]
                if hist_idx.size > 0
                else np.empty(0, dtype=np.int64)
            )

            unique_cards = np.unique(np.concatenate([target_cards_np, hist_cards_np], axis=0))
            global_to_local = {int(c): i for i, c in enumerate(unique_cards.tolist())}

            target_card_local = np.array(
                [global_to_local[int(c)] for c in target_cards_np], dtype=np.int64
            )
            hist_card_local = (
                np.array([global_to_local[int(c)] for c in hist_cards_np], dtype=np.int64)
                if hist_cards_np.size > 0
                else np.empty(0, dtype=np.int64)
            )

            card_feats_np = stats.get_features(
                unique_cards, np.zeros(len(unique_cards), dtype=np.float32)
            )

            optimizer.zero_grad()

            logits = model(
                target_x=torch.tensor(target_x_np, dtype=torch.float32, device=device),
                hist_x=torch.tensor(hist_x_np, dtype=torch.float32, device=device),
                hist_card_local_idx=torch.tensor(hist_card_local, dtype=torch.long, device=device),
                target_card_local_idx=torch.tensor(
                    target_card_local, dtype=torch.long, device=device
                ),
                card_dense_feats=torch.tensor(card_feats_np, dtype=torch.float32, device=device),
            )
            y = torch.tensor(target_y_np, dtype=torch.float32, device=device)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))

            stats.update(
                card_ids=target_cards_np,
                labels=target_y_np,
                amt_values=target_x_np[:, amt_idx],
            )

        val_pred = predict_gnn_strict(
            model=model,
            history_splits=[train_split],
            target_split=val_split,
            n_cards=n_cards,
            feature_columns=feature_columns,
            device=device,
            batch_size=batch_size,
            max_hist_per_card=max_hist_per_card,
        )
        val_metrics = compute_metrics(val_split.y, val_pred)
        print(
            f"[OOF-GNN] epoch={epoch} loss={np.mean(losses):.5f} "
            f"val_pr_auc={val_metrics['pr_auc']:.5f}"
        )

        if val_metrics["pr_auc"] > best_val_pr:
            best_val_pr = val_metrics["pr_auc"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No best_state captured during OOF GNN training.")

    model.load_state_dict(best_state)
    model.eval()
    return model


def build_train_oof_predictions(
    train_split: SplitData,
    *,
    n_cards: int,
    feature_columns: list[str],
    device: torch.device,
    n_blocks: int = 6,
    min_history_blocks: int = 2,
    hidden_dim: int = 128,
    dropout: float = 0.2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 4,
    batch_size: int = 2048,
    max_hist_per_card: int = 50,
) -> np.ndarray:
    blocks = make_blocks(train_split.n, n_blocks)
    oof = np.full(train_split.n, np.nan, dtype=np.float32)

    for block_id in range(min_history_blocks, n_blocks):
        pred_start, pred_end = blocks[block_id]
        hist_end = pred_start

        hist_split = make_split(
            train_split.X[:hist_end],
            train_split.y[:hist_end],
            train_split.card_idx[:hist_end],
            train_split.time[:hist_end],
        )
        target_split = make_split(
            train_split.X[pred_start:pred_end],
            train_split.y[pred_start:pred_end],
            train_split.card_idx[pred_start:pred_end],
            train_split.time[pred_start:pred_end],
        )

        print(f"[OOF] block={block_id} history=[0:{hist_end}] target=[{pred_start}:{pred_end}]")

        model = fit_gnn_on_history(
            hist_split,
            target_split,
            n_cards=n_cards,
            feature_columns=feature_columns,
            device=device,
            hidden_dim=hidden_dim,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            max_hist_per_card=max_hist_per_card,
        )

        pred = predict_gnn_strict(
            model=model,
            history_splits=[hist_split],
            target_split=target_split,
            n_cards=n_cards,
            feature_columns=feature_columns,
            device=device,
            batch_size=batch_size,
            max_hist_per_card=max_hist_per_card,
        )

        oof[pred_start:pred_end] = np.asarray(pred, dtype=np.float32)

    return oof


def load_final_model(
    txn_in_dim: int,
    *,
    device: torch.device,
    hidden_dim: int = 128,
    dropout: float = 0.2,
) -> CardHistorySAGE:
    model = CardHistorySAGE(
        txn_in_dim=txn_in_dim,
        card_feat_dim=5,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    state = torch.load(PHASE4_MODEL_DIR / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def save_prediction_table(split_name: str, scores: np.ndarray, prediction_kind: str) -> None:
    df = pd.read_parquet(PROCESSED_DIR / f"{split_name}_df.parquet").reset_index(drop=True)

    out = pd.DataFrame(
        {
            "row_idx": np.arange(len(df), dtype=np.int64),
            "gnn_score": scores.astype(np.float32),
            "prediction_kind": prediction_kind,
        }
    )

    if "TransactionID" in df.columns:
        out["TransactionID"] = df["TransactionID"].values

    cols = ["row_idx"]
    if "TransactionID" in out.columns:
        cols.append("TransactionID")
    cols += ["gnn_score", "prediction_kind"]

    out[cols].to_parquet(PHASE4_PRED_DIR / f"{split_name}_predictions.parquet", index=False)


def main() -> None:
    PHASE4_PRED_DIR.mkdir(parents=True, exist_ok=True)

    arr = load_arrays()
    meta = arr["metadata"]

    feature_columns = meta["feature_columns"]
    n_cards = int(meta["n_cards"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_split = make_split(
        arr["X_train"], arr["y_train"], arr["card_idx_train"], arr["time_train"]
    )
    val_split = make_split(
        arr["X_val"], arr["y_val"], arr["card_idx_val"], arr["time_val"]
    )
    test_split = make_split(
        arr["X_test"], arr["y_test"], arr["card_idx_test"], arr["time_test"]
    )

    train_oof_pred = build_train_oof_predictions(
        train_split=train_split,
        n_cards=n_cards,
        feature_columns=feature_columns,
        device=device,
    )
    np.save(PHASE4_PRED_DIR / "train_oof_pred.npy", train_oof_pred)

    final_model = load_final_model(
        txn_in_dim=train_split.X.shape[1],
        device=device,
    )

    val_pred = predict_gnn_strict(
        model=final_model,
        history_splits=[train_split],
        target_split=val_split,
        n_cards=n_cards,
        feature_columns=feature_columns,
        device=device,
    ).astype(np.float32)
    np.save(PHASE4_PRED_DIR / "val_pred.npy", val_pred)

    test_pred = predict_gnn_strict(
        model=final_model,
        history_splits=[train_split, val_split],
        target_split=test_split,
        n_cards=n_cards,
        feature_columns=feature_columns,
        device=device,
    ).astype(np.float32)
    np.save(PHASE4_PRED_DIR / "test_pred.npy", test_pred)

    save_prediction_table("train", train_oof_pred, "train_oof")
    save_prediction_table("val", val_pred, "strict_eval")
    save_prediction_table("test", test_pred, "strict_eval")

    print("\nSaved:")
    print(PHASE4_PRED_DIR / "train_oof_pred.npy")
    print(PHASE4_PRED_DIR / "val_pred.npy")
    print(PHASE4_PRED_DIR / "test_pred.npy")
    print(PHASE4_PRED_DIR / "train_predictions.parquet")
    print(PHASE4_PRED_DIR / "val_predictions.parquet")
    print(PHASE4_PRED_DIR / "test_predictions.parquet")


if __name__ == "__main__":
    main()
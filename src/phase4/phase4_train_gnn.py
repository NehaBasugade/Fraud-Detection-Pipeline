from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.phase4.phase4_dataset import (
    RunningCardStats,
    SplitData,
    build_card_history_index,
    get_recent_history_for_cards,
    make_chronological_batches,
)
from src.phase4.phase4_evaluate import compute_metrics, save_metrics
from src.phase4.phase4_infer import predict_gnn_strict
from src.phase4.phase4_models import CardHistorySAGE


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "artifacts" / "phase4" / "data"
MODEL_DIR = ROOT / "artifacts" / "phase4" / "models" / "baseline_gnn_card_only"
PRED_DIR = ROOT / "artifacts" / "phase4" / "predictions" / "baseline_gnn_card_only"
REPORT_DIR = ROOT / "reports" / "phase4" / "baseline_gnn_card_only"


def load_metadata() -> dict:
    with open(DATA_DIR / "metadata.json", "r") as f:
        return json.load(f)


def load_split(name: str) -> SplitData:
    return SplitData(
        X=np.load(DATA_DIR / f"X_{name}.npy"),
        y=np.load(DATA_DIR / f"y_{name}.npy"),
        card_idx=np.load(DATA_DIR / f"card_idx_{name}.npy"),
        time=np.load(DATA_DIR / f"time_{name}.npy"),
    ).sorted_view()


def amt_column_index(feature_columns: list[str]) -> int:
    for c in ["TransactionAmt", "TransactionAMT", "transactionamt"]:
        if c in feature_columns:
            return feature_columns.index(c)
    raise ValueError("TransactionAmt not found in feature columns")


def train_gnn(
    hidden_dim: int = 128,
    dropout: float = 0.2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 4,
    batch_size: int = 2048,
    max_hist_per_card: int = 50,
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    meta = load_metadata()
    feature_columns = meta["feature_columns"]
    n_cards = int(meta["n_cards"])
    amt_idx = amt_column_index(feature_columns)

    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CardHistorySAGE(
        txn_in_dim=train.X.shape[1],
        card_feat_dim=5,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    pos_weight = (len(train.y) - train.y.sum()) / max(train.y.sum(), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    card_hist_idx = build_card_history_index(train.card_idx)

    best_val_pr = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        stats = RunningCardStats(n_cards=n_cards)

        for batch in make_chronological_batches(train.n, batch_size):
            target_idx = batch.target_idx
            history_idx = batch.history_idx

            target_x_np = train.X[target_idx]
            target_y_np = train.y[target_idx]
            target_cards_np = train.card_idx[target_idx]

            hist_idx = get_recent_history_for_cards(
                target_cards=target_cards_np,
                history_pool_idx=history_idx,
                card_history_index=card_hist_idx,
                max_hist_per_card=max_hist_per_card,
            )

            hist_x_np = train.X[hist_idx] if hist_idx.size > 0 else np.empty((0, train.X.shape[1]), dtype=np.float32)
            hist_cards_np = train.card_idx[hist_idx] if hist_idx.size > 0 else np.empty(0, dtype=np.int64)

            unique_cards = np.unique(np.concatenate([target_cards_np, hist_cards_np], axis=0))
            global_to_local = {int(c): i for i, c in enumerate(unique_cards.tolist())}

            target_card_local = np.array([global_to_local[int(c)] for c in target_cards_np], dtype=np.int64)
            hist_card_local = (
                np.array([global_to_local[int(c)] for c in hist_cards_np], dtype=np.int64)
                if hist_cards_np.size > 0
                else np.empty(0, dtype=np.int64)
            )

            card_feats_np = stats.get_features(unique_cards, np.zeros(len(unique_cards), dtype=np.float32))

            optimizer.zero_grad()

            logits = model(
                target_x=torch.tensor(target_x_np, dtype=torch.float32, device=device),
                hist_x=torch.tensor(hist_x_np, dtype=torch.float32, device=device),
                hist_card_local_idx=torch.tensor(hist_card_local, dtype=torch.long, device=device),
                target_card_local_idx=torch.tensor(target_card_local, dtype=torch.long, device=device),
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
            history_splits=[train],
            target_split=val,
            n_cards=n_cards,
            feature_columns=feature_columns,
            device=device,
            batch_size=batch_size,
            max_hist_per_card=max_hist_per_card,
        )
        val_metrics = compute_metrics(val.y, val_pred)
        print(f"[GNN] epoch={epoch} loss={np.mean(losses):.5f} val_pr_auc={val_metrics['pr_auc']:.5f}")

        if val_metrics["pr_auc"] > best_val_pr:
            best_val_pr = val_metrics["pr_auc"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    torch.save(best_state, MODEL_DIR / "model.pt")

    model.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location=device))
    val_pred = predict_gnn_strict(
        model=model,
        history_splits=[train],
        target_split=val,
        n_cards=n_cards,
        feature_columns=feature_columns,
        device=device,
        batch_size=batch_size,
        max_hist_per_card=max_hist_per_card,
    )
    test_pred = predict_gnn_strict(
        model=model,
        history_splits=[train, val],
        target_split=test,
        n_cards=n_cards,
        feature_columns=feature_columns,
        device=device,
        batch_size=batch_size,
        max_hist_per_card=max_hist_per_card,
    )

    np.save(PRED_DIR / "val_pred.npy", val_pred)
    np.save(PRED_DIR / "test_pred.npy", test_pred)

    val_metrics = compute_metrics(val.y, val_pred)
    test_metrics = compute_metrics(test.y, test_pred)
    save_metrics(REPORT_DIR, "val", val_metrics)
    save_metrics(REPORT_DIR, "test", test_metrics)

    with open(REPORT_DIR / "train_config.json", "w") as f:
        json.dump(
            {
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "lr": lr,
                "weight_decay": weight_decay,
                "epochs": epochs,
                "batch_size": batch_size,
                "max_hist_per_card": max_hist_per_card,
                "pos_weight": float(pos_weight),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    train_gnn()
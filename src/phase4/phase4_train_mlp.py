from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.phase4.phase4_dataset import SplitData, make_chronological_batches
from src.phase4.phase4_evaluate import compute_metrics, save_metrics
from src.phase4.phase4_infer import predict_mlp
from src.phase4.phase4_models import MLPFraud


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "artifacts" / "phase4" / "data"
MODEL_DIR = ROOT / "artifacts" / "phase4" / "models" / "baseline_mlp"
PRED_DIR = ROOT / "artifacts" / "phase4" / "predictions" / "baseline_mlp"
REPORT_DIR = ROOT / "reports" / "phase4" / "baseline_mlp"


def load_split(name: str) -> SplitData:
    return SplitData(
        X=np.load(DATA_DIR / f"X_{name}.npy"),
        y=np.load(DATA_DIR / f"y_{name}.npy"),
        card_idx=np.load(DATA_DIR / f"card_idx_{name}.npy"),
        time=np.load(DATA_DIR / f"time_{name}.npy"),
    ).sorted_view()


def train_mlp(
    hidden_dim: int = 256,
    dropout: float = 0.2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 5,
    batch_size: int = 4096,
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLPFraud(in_dim=train.X.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
    pos_weight = (len(train.y) - train.y.sum()) / max(train.y.sum(), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_pr = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for batch in make_chronological_batches(train.n, batch_size):
            x = torch.tensor(train.X[batch.target_idx], dtype=torch.float32, device=device)
            y = torch.tensor(train.y[batch.target_idx], dtype=torch.float32, device=device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        val_pred = predict_mlp(model, val, device=device, batch_size=batch_size)
        val_metrics = compute_metrics(val.y, val_pred)
        print(f"[MLP] epoch={epoch} loss={np.mean(losses):.5f} val_pr_auc={val_metrics['pr_auc']:.5f}")

        if val_metrics["pr_auc"] > best_val_pr:
            best_val_pr = val_metrics["pr_auc"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    torch.save(best_state, MODEL_DIR / "model.pt")

    model.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location=device))
    val_pred = predict_mlp(model, val, device=device, batch_size=batch_size)
    test_pred = predict_mlp(model, test, device=device, batch_size=batch_size)

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
                "pos_weight": float(pos_weight),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    train_mlp()
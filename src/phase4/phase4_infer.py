from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.phase4.phase4_dataset import (
    RunningCardStats,
    SplitData,
    build_card_history_index,
    get_recent_history_for_cards,
    make_chronological_batches,
)
from src.phase4.phase4_models import CardHistorySAGE, MLPFraud


def _amt_column_index(feature_columns: list[str]) -> int:
    for c in ["TransactionAmt", "TransactionAMT", "transactionamt"]:
        if c in feature_columns:
            return feature_columns.index(c)
    raise ValueError("TransactionAmt column not found in metadata feature_columns")


@torch.no_grad()
def predict_mlp(
    model: MLPFraud,
    split: SplitData,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    preds = []
    for batch in make_chronological_batches(split.n, batch_size):
        x = torch.tensor(split.X[batch.target_idx], dtype=torch.float32, device=device)
        logits = model(x)
        preds.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(preds, axis=0)


@torch.no_grad()
def predict_gnn_strict(
    model: CardHistorySAGE,
    history_splits: list[SplitData],
    target_split: SplitData,
    n_cards: int,
    feature_columns: list[str],
    device: torch.device,
    batch_size: int = 2048,
    max_hist_per_card: int = 50,
) -> np.ndarray:
    model.eval()
    amt_idx = _amt_column_index(feature_columns)

    combined_hist = None
    if history_splits:
        X_hist = np.concatenate([s.X for s in history_splits], axis=0)
        y_hist = np.concatenate([s.y for s in history_splits], axis=0)
        c_hist = np.concatenate([s.card_idx for s in history_splits], axis=0)
        t_hist = np.concatenate([s.time for s in history_splits], axis=0)
        combined_hist = SplitData(X_hist, y_hist, c_hist, t_hist).sorted_view()

    target_split = target_split.sorted_view()

    if combined_hist is None:
        hist_X = np.empty((0, target_split.X.shape[1]), dtype=np.float32)
        hist_y = np.empty(0, dtype=np.float32)
        hist_c = np.empty(0, dtype=np.int64)
    else:
        hist_X, hist_y, hist_c = combined_hist.X, combined_hist.y, combined_hist.card_idx

    stats = RunningCardStats(n_cards=n_cards)
    if hist_c.size > 0:
        stats.update(hist_c, hist_y, hist_X[:, amt_idx])

    preds = []
    target_card_hist_idx = build_card_history_index(target_split.card_idx)

    for batch in make_chronological_batches(target_split.n, batch_size):
        target_idx = batch.target_idx
        target_x = target_split.X[target_idx]
        target_cards = target_split.card_idx[target_idx]

        hist_local_idx = get_recent_history_for_cards(
            target_cards=target_cards,
            history_pool_idx=batch.history_idx,
            card_history_index=target_card_hist_idx,
            max_hist_per_card=max_hist_per_card,
        )

        if hist_local_idx.size > 0:
            cur_hist_x = np.concatenate([hist_X, target_split.X[hist_local_idx]], axis=0)
            cur_hist_c = np.concatenate([hist_c, target_split.card_idx[hist_local_idx]], axis=0)
        else:
            cur_hist_x = hist_X
            cur_hist_c = hist_c

        unique_cards = np.unique(np.concatenate([target_cards, cur_hist_c], axis=0))
        global_to_local = {int(c): i for i, c in enumerate(unique_cards.tolist())}

        target_card_local = np.array([global_to_local[int(c)] for c in target_cards], dtype=np.int64)
        if cur_hist_c.size > 0:
            hist_card_local = np.array([global_to_local[int(c)] for c in cur_hist_c], dtype=np.int64)
        else:
            hist_card_local = np.empty(0, dtype=np.int64)

        card_feats = stats.get_features(unique_cards, np.zeros(len(unique_cards), dtype=np.float32))

        logits = model(
            target_x=torch.tensor(target_x, dtype=torch.float32, device=device),
            hist_x=torch.tensor(cur_hist_x, dtype=torch.float32, device=device),
            hist_card_local_idx=torch.tensor(hist_card_local, dtype=torch.long, device=device),
            target_card_local_idx=torch.tensor(target_card_local, dtype=torch.long, device=device),
            card_dense_feats=torch.tensor(card_feats, dtype=torch.float32, device=device),
        )
        preds.append(torch.sigmoid(logits).cpu().numpy())

        stats.update(
            card_ids=target_cards,
            labels=target_split.y[target_idx],
            amt_values=target_x[:, amt_idx],
        )

    return np.concatenate(preds, axis=0)
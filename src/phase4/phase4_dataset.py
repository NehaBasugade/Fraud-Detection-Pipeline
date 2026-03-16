from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class SplitData:
    X: np.ndarray
    y: np.ndarray
    card_idx: np.ndarray
    time: np.ndarray

    @property
    def n(self) -> int:
        return len(self.y)

    def sorted_view(self) -> "SplitData":
        order = np.argsort(self.time, kind="stable")
        return SplitData(
            X=self.X[order],
            y=self.y[order],
            card_idx=self.card_idx[order],
            time=self.time[order],
        )


@dataclass
class Batch:
    target_idx: np.ndarray
    history_idx: np.ndarray


def make_chronological_batches(n: int, batch_size: int) -> Iterator[Batch]:
    start = 0
    while start < n:
        end = min(start + batch_size, n)
        yield Batch(
            target_idx=np.arange(start, end, dtype=np.int64),
            history_idx=np.arange(0, start, dtype=np.int64),
        )
        start = end


def build_card_history_index(card_idx: np.ndarray) -> dict[int, np.ndarray]:
    unique_cards = np.unique(card_idx)
    return {int(c): np.flatnonzero(card_idx == c).astype(np.int64) for c in unique_cards}


def get_recent_history_for_cards(
    target_cards: np.ndarray,
    history_pool_idx: np.ndarray,
    card_history_index: dict[int, np.ndarray],
    max_hist_per_card: int,
) -> np.ndarray:
    if history_pool_idx.size == 0:
        return np.empty(0, dtype=np.int64)

    history_end = int(history_pool_idx[-1]) + 1
    out = []
    seen = set()

    for c in np.unique(target_cards):
        card_hist = card_history_index.get(int(c))
        if card_hist is None or card_hist.size == 0:
            continue

        cut = np.searchsorted(card_hist, history_end, side="left")
        selected = card_hist[max(0, cut - max_hist_per_card):cut]
        for idx in selected:
            idx = int(idx)
            if idx not in seen:
                seen.add(idx)
                out.append(idx)

    if not out:
        return np.empty(0, dtype=np.int64)
    return np.array(sorted(out), dtype=np.int64)


class RunningCardStats:
    def __init__(self, n_cards: int, smoothing: float = 20.0):
        self.n_cards = n_cards
        self.smoothing = smoothing
        self.count = np.zeros(n_cards, dtype=np.float32)
        self.fraud = np.zeros(n_cards, dtype=np.float32)
        self.amt_sum = np.zeros(n_cards, dtype=np.float32)
        self.amt_sq_sum = np.zeros(n_cards, dtype=np.float32)

    def get_features(self, card_ids: np.ndarray, amt_values: np.ndarray) -> np.ndarray:
        cnt = self.count[card_ids]
        frd = self.fraud[card_ids]
        amt_mean = np.divide(self.amt_sum[card_ids], cnt, out=np.zeros_like(cnt), where=cnt > 0)
        amt_var = np.divide(self.amt_sq_sum[card_ids], cnt, out=np.zeros_like(cnt), where=cnt > 0) - (amt_mean ** 2)
        amt_std = np.sqrt(np.clip(amt_var, 0.0, None))
        global_rate = float(self.fraud.sum() / max(self.count.sum(), 1.0))
        smoothed_rate = (frd + self.smoothing * global_rate) / (cnt + self.smoothing)
        return np.stack([cnt, frd, smoothed_rate, amt_mean, amt_std], axis=1).astype(np.float32)

    def update(self, card_ids: np.ndarray, labels: np.ndarray, amt_values: np.ndarray) -> None:
        np.add.at(self.count, card_ids, 1.0)
        np.add.at(self.fraud, card_ids, labels.astype(np.float32))
        np.add.at(self.amt_sum, card_ids, amt_values.astype(np.float32))
        np.add.at(self.amt_sq_sum, card_ids, np.square(amt_values.astype(np.float32)))
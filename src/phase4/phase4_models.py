from __future__ import annotations

import torch
import torch.nn as nn


class MLPFraud(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CardHistorySAGE(nn.Module):
    """
    Strict baseline:
    - target transactions NEVER message to each other
    - only prior history transactions -> cards -> target transactions
    """

    def __init__(
        self,
        txn_in_dim: int,
        card_feat_dim: int = 5,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.txn_encoder = nn.Sequential(
            nn.Linear(txn_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.card_mlp = nn.Sequential(
            nn.Linear(hidden_dim + card_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        target_x: torch.Tensor,
        hist_x: torch.Tensor,
        hist_card_local_idx: torch.Tensor,
        target_card_local_idx: torch.Tensor,
        card_dense_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        target_x: [B, F]
        hist_x: [H, F]
        hist_card_local_idx: [H] values in [0, U_cards)
        target_card_local_idx: [B] values in [0, U_cards)
        card_dense_feats: [U_cards, card_feat_dim]
        """
        target_h = self.txn_encoder(target_x)

        if hist_x.numel() == 0:
            hist_agg = torch.zeros(
                (card_dense_feats.size(0), target_h.size(1)),
                dtype=target_h.dtype,
                device=target_h.device,
            )
        else:
            hist_h = self.txn_encoder(hist_x)
            n_cards = card_dense_feats.size(0)
            hidden_dim = hist_h.size(1)

            hist_sum = torch.zeros((n_cards, hidden_dim), device=hist_h.device, dtype=hist_h.dtype)
            hist_cnt = torch.zeros((n_cards, 1), device=hist_h.device, dtype=hist_h.dtype)

            hist_sum.index_add_(0, hist_card_local_idx, hist_h)
            ones = torch.ones((hist_h.size(0), 1), device=hist_h.device, dtype=hist_h.dtype)
            hist_cnt.index_add_(0, hist_card_local_idx, ones)
            hist_agg = hist_sum / hist_cnt.clamp_min(1.0)

        card_h = self.card_mlp(torch.cat([card_dense_feats, hist_agg], dim=1))
        target_card_h = card_h[target_card_local_idx]
        logits = self.head(torch.cat([target_h, target_card_h], dim=1)).squeeze(-1)
        return logits
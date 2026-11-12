"""Bidirectional LSTM regressor for next-hour soil moisture."""
from __future__ import annotations

import torch
from torch import nn


class BiLSTMRegressor(nn.Module):
    """Stacked bidirectional LSTM over the 24-h input window.

    The forward direction's final state and the backward direction's final
    state (which has read the window from the latest hour back to the first)
    are concatenated and passed to a linear head. Bidirectionality is only
    over the *past* input window, so no future information is used.
    """

    def __init__(self, n_features: int = 11, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        # h: (num_layers * 2, batch, hidden); last layer's forward and backward states
        last = torch.cat([h[-2], h[-1]], dim=-1)
        return self.head(last).squeeze(-1)

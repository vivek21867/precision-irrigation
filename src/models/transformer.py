"""Transformer-encoder regressor for next-hour soil moisture."""
from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerRegressor(nn.Module):
    """Input projection -> positional encoding -> N x (multi-head self-attention +
    feed-forward, each with residual + LayerNorm + dropout) -> final LayerNorm ->
    last-timestep representation -> linear regression head.
    """

    def __init__(self, n_features: int = 11, d_model: int = 64, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 128, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.dropout(self.pos(self.input_proj(x)))
        z = self.norm(self.encoder(z))
        return self.head(z[:, -1, :]).squeeze(-1)

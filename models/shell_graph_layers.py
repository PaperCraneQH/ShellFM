"""Shared building blocks for the Shell-Graph Graph-Transformer backbone."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .shell_graph_model import _build_chain_k_adjacency


class LSTMNodeEmbed(nn.Module):
    """Bi-LSTM node embedding: [B, N, in_dim] -> [B, N, hidden_dim]."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % 2 == 0, 'hidden_dim must be even for bi-LSTM'
        self.num_layers = int(num_layers)
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim // 2,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=(dropout if self.num_layers > 1 else 0.0),
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        h = self.ln(h)
        return self.dropout(h)


class PositionalEncoding(nn.Module):
    """Positional / structural encoding along the shell axis (length N)."""

    def __init__(self, mode: str, n_shells: int, hidden_dim: int,
                 k_neighbors: int, k_rw: int = 8):
        super().__init__()
        assert mode in ('none', 'sinusoidal', 'learnable', 'rwse'), \
            f'unknown PE mode: {mode}'
        self.mode = mode
        self.n_shells = int(n_shells)
        self.hidden_dim = int(hidden_dim)

        if mode == 'sinusoidal':
            pe = torch.zeros(n_shells, hidden_dim)
            position = torch.arange(0, n_shells, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, hidden_dim, 2).float()
                * (-math.log(10000.0) / hidden_dim)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe, persistent=False)

        elif mode == 'learnable':
            self.embed = nn.Embedding(n_shells, hidden_dim)
            nn.init.normal_(self.embed.weight, std=0.02)

        elif mode == 'rwse':
            adj = _build_chain_k_adjacency(n_shells, k_neighbors)
            A = adj['A_self'].float()
            deg = A.sum(dim=-1).clamp(min=1.0)
            P = A / deg.unsqueeze(-1)
            P_t = P.clone()
            rwse_diags = []
            for _ in range(k_rw):
                rwse_diags.append(torch.diagonal(P_t, 0))
                P_t = P_t @ P
            rwse = torch.stack(rwse_diags, dim=-1)
            self.register_buffer('rwse_features', rwse, persistent=False)
            self.proj = nn.Linear(k_rw, hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.mode == 'none':
            return h
        if self.mode == 'sinusoidal':
            return h + self.pe.unsqueeze(0)
        if self.mode == 'learnable':
            idx = torch.arange(self.n_shells, device=h.device)
            return h + self.embed(idx).unsqueeze(0)
        if self.mode == 'rwse':
            pe = self.proj(self.rwse_features)
            return h + pe.unsqueeze(0)
        return h


class GraphPooling(nn.Module):
    """Graph-level readout: [B, N, D] -> [B, out_dim]."""

    def __init__(self, mode: str, n_shells: int, hidden_dim: int):
        super().__init__()
        assert mode in ('flatten', 'mean', 'sum', 'max', 'maxmean', 'attn'), \
            f'unknown pooling mode: {mode}'
        self.mode = mode
        self.n_shells = int(n_shells)
        self.hidden_dim = int(hidden_dim)

        if mode == 'flatten':
            self.out_dim = self.n_shells * self.hidden_dim
        elif mode in ('mean', 'sum', 'max'):
            self.out_dim = self.hidden_dim
        elif mode == 'maxmean':
            self.out_dim = 2 * self.hidden_dim
        elif mode == 'attn':
            self.out_dim = self.hidden_dim
            self.q_proj = nn.Linear(hidden_dim, hidden_dim)
            self.attn_query = nn.Parameter(torch.randn(hidden_dim) * 0.02)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.mode == 'flatten':
            return h.reshape(h.size(0), -1)
        if self.mode == 'mean':
            return h.mean(dim=1)
        if self.mode == 'sum':
            return h.sum(dim=1)
        if self.mode == 'max':
            return h.max(dim=1).values
        if self.mode == 'maxmean':
            return torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)
        if self.mode == 'attn':
            scores = (self.q_proj(h) * self.attn_query).sum(dim=-1)
            weights = F.softmax(scores, dim=-1).unsqueeze(-1)
            return (weights * h).sum(dim=1)
        raise RuntimeError(f'unsupported pooling mode {self.mode}')

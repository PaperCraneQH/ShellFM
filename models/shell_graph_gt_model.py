"""Shell-Graph Graph-Transformer backbone (Transformer block with graph attention inside).

Each block follows MAGNA-style Pre-LN + residual updates: graph attention is the sole
token-mixing operator on chain-k shell neighbors, followed by a feed-forward sub-layer.
Supported aggregation schemes: GAT, GCN, GIN, and alternating GAT_GCN.

Used by ``ShellGraphStructEncoder`` in ``struct_tower.py`` as the structure-tower backbone.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .shell_graph_model import _build_chain_k_adjacency
from .shell_graph_layers import LSTMNodeEmbed, PositionalEncoding, GraphPooling


class GraphAttention(nn.Module):
    """Single graph-attention operator: attention/aggregation over chain-k adjacency.

    Core mechanism that **fuses attention into graph node updates** — it acts as both
    GNN graph aggregation and Transformer token-mixing. Three schemes set w_ij:
      * GAT: learnable multi-head scaled dot-product attention, softmax over graph neighbors;
      * GCN: fixed symmetric normalized adjacency Â (no learnable attention);
      * GIN: (1+ε) self-weight + neighbor sum.
    All three use multi-head W_V + W_O output projection to fit the outer Transformer block.
    """

    SCHEMES = ('GAT', 'GCN', 'GIN')

    def __init__(self, hidden_dim: int, n_heads: int, scheme: str, dropout: float = 0.1):
        super().__init__()
        scheme = scheme.upper()
        assert scheme in self.SCHEMES, f'scheme must be one of {self.SCHEMES}, got {scheme!r}'
        assert hidden_dim % n_heads == 0, \
            f'hidden_dim={hidden_dim} must be divisible by n_heads={n_heads}'
        self.scheme = scheme
        self.n_heads = int(n_heads)
        self.head_dim = hidden_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.proj_dropout = nn.Dropout(dropout)

        if scheme == 'GAT':
            self.q_proj = nn.Linear(hidden_dim, hidden_dim)
            self.k_proj = nn.Linear(hidden_dim, hidden_dim)
            self.attn_dropout = nn.Dropout(dropout)
        if scheme == 'GIN':
            self.eps = nn.Parameter(torch.zeros(1))

    def _split_heads(self, t: torch.Tensor, B: int, N: int) -> torch.Tensor:
        # [B, N, D] -> [B, H, N, head_dim]
        return t.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, A_self: torch.Tensor,
                A_kipf: torch.Tensor, A_raw: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D] (LayerNorm-ed); A_*: [N, N] graph adjacency (chain-k)
        B, N, D = x.shape
        V = self._split_heads(self.v_proj(x), B, N)            # [B, H, N, hd]

        if self.scheme == 'GAT':
            Q = self._split_heads(self.q_proj(x), B, N)
            K = self._split_heads(self.k_proj(x), B, N)
            e = (Q @ K.transpose(-2, -1)) * self.scale         # [B, H, N, N]
            # Only graph neighbors (incl. self-loops) attend -> attention follows graph structure
            mask = (A_self == 0).unsqueeze(0).unsqueeze(0)      # [1, 1, N, N]
            e = e.masked_fill(mask, float('-inf'))
            alpha = F.softmax(e, dim=-1)
            alpha = self.attn_dropout(alpha)
            out = alpha @ V                                    # [B, H, N, hd]
        elif self.scheme == 'GCN':
            # Fixed symmetric normalized aggregation: out = Â V (Â includes self-loops, normalized)
            out = torch.einsum('ij,bhjd->bhid', A_kipf, V)
        else:  # GIN
            neigh = torch.einsum('ij,bhjd->bhid', A_raw, V)    # neighbor sum (A_raw has no self-loops)
            out = (1.0 + self.eps) * V + neigh

        out = out.transpose(1, 2).reshape(B, N, D)             # [B, N, D]
        return self.proj_dropout(self.out_proj(out))


class GraphTransformerBlock(nn.Module):
    """Standard Transformer block with graph attention as the attention sub-layer.

    Sub-layers (all Pre-LN + residual):
        x = x + GraphAttention(LN(x), graph adjacency)   # attention = graph update
        x = x + FFN(LN(x))                               # feed-forward (deep aggregation)
    """

    def __init__(self, hidden_dim: int, n_heads: int, scheme: str, dropout: float = 0.1,
                 use_ffn: bool = True, ffn_mult: int = 2):
        super().__init__()
        self.attn_ln = nn.LayerNorm(hidden_dim)
        self.attn = GraphAttention(hidden_dim, n_heads, scheme, dropout)
        self.use_ffn = bool(use_ffn)
        if self.use_ffn:
            self.ffn_ln = nn.LayerNorm(hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, ffn_mult * hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_mult * hidden_dim, hidden_dim),
                nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor, A_self: torch.Tensor,
                A_kipf: torch.Tensor, A_raw: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_ln(x), A_self, A_kipf, A_raw)
        if self.use_ffn:
            x = x + self.ffn(self.ffn_ln(x))
        return x


class ShellGraphGTModel(nn.Module):
    """Shell-Graph Graph-Transformer: LSTM node embed + (graph-attention Transformer)×L + head.

    Args:
        in_shape: (n_pairs, n_shells, channel)
        hidden_dim: node hidden dim (default 128; must be even for bi-LSTM)
        n_layers: number of GraphTransformerBlock layers (default 3)
        k_neighbors: chain-k graph adjacency radius (default 2; attention over this graph)
        gnn_type: 'GCN' | 'GIN' | 'GAT' | 'GAT_GCN' — graph-attention aggregation scheme
        n_heads: number of graph-attention heads (default 4)
        use_ffn: enable FFN sub-layer (default True, standard Transformer block)
        ffn_mult: FFN hidden-dim multiplier (default 2)
        attn_dropout: attention/FFN sub-layer dropout (default follows dropout)
        pos_encoding / pooling: same as PARTX
    """

    GNN_TYPES = ('GCN', 'GIN', 'GAT', 'GAT_GCN')

    def __init__(self,
                 in_shape: Tuple[int, int, int] = (168, 60, 1),
                 hidden_dim: int = 128,
                 n_layers: int = 3,
                 k_neighbors: int = 2,
                 gnn_type: str = 'GAT_GCN',
                 n_heads: int = 4,
                 dropout: float = 0.1,
                 lstm_num_layers: int = 1,
                 use_ffn: bool = True,
                 ffn_mult: int = 2,
                 attn_dropout: float = -1.0,
                 pos_encoding: str = 'none',
                 pos_encoding_kdim: int = 8,
                 pooling: str = 'flatten'):
        super().__init__()
        gnn_type = gnn_type.upper()
        assert gnn_type in self.GNN_TYPES, \
            f'gnn_type must be one of {self.GNN_TYPES}, got {gnn_type!r}'
        n_pairs, n_shells, _ = in_shape
        self.gnn_type = gnn_type
        self.n_pairs = int(n_pairs)
        self.n_shells = int(n_shells)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.k_neighbors = int(k_neighbors)
        self.n_heads = int(n_heads)
        self.lstm_num_layers = int(lstm_num_layers)
        self.use_ffn = bool(use_ffn)
        self.ffn_mult = int(ffn_mult)
        self.attn_dropout = float(attn_dropout) if float(attn_dropout) >= 0 else float(dropout)
        self.pos_encoding = pos_encoding
        self.pos_encoding_kdim = int(pos_encoding_kdim)
        self.pooling = pooling
        assert self.hidden_dim % 2 == 0, 'hidden_dim must be even (for bi-LSTM)'

        # ---- Stage 1: bi-LSTM node embed ----
        self.node_proj = LSTMNodeEmbed(self.n_pairs, self.hidden_dim,
                                       num_layers=self.lstm_num_layers, dropout=dropout)

        # ---- Stage 1.5: positional / structural encoding ----
        self.pe = PositionalEncoding(
            mode=self.pos_encoding,
            n_shells=self.n_shells,
            hidden_dim=self.hidden_dim,
            k_neighbors=self.k_neighbors,
            k_rw=self.pos_encoding_kdim,
        )

        # ---- Graph adjacency (chain-k); attention/aggregation follows this structure ----
        adj = _build_chain_k_adjacency(self.n_shells, self.k_neighbors)
        self.register_buffer('A_raw', adj['A_raw'], persistent=False)
        self.register_buffer('A_self', adj['A_self'], persistent=False)
        self.register_buffer('A_kipf', adj['A_kipf'], persistent=False)
        self._avg_degree = float(adj['A_raw'].sum(dim=1).mean().item())

        # ---- Per-layer scheme (GAT_GCN alternates between layers) ----
        self.schemes: List[str] = []
        for li in range(self.n_layers):
            if gnn_type == 'GAT_GCN':
                self.schemes.append('GAT' if li % 2 == 0 else 'GCN')
            else:
                self.schemes.append(gnn_type)

        # ---- Stage 2: GraphTransformerBlock × n_layers ----
        self.blocks = nn.ModuleList([
            GraphTransformerBlock(
                self.hidden_dim, self.n_heads, scheme,
                dropout=self.attn_dropout, use_ffn=self.use_ffn, ffn_mult=self.ffn_mult,
            ) for scheme in self.schemes
        ])

        # ---- Stage 3: pooling + dense head (same as PARTX; no Dropout in head) ----
        self.pool = GraphPooling(self.pooling, self.n_shells, self.hidden_dim)
        head_in = self.pool.out_dim
        self.classifier = nn.Sequential(
            nn.Linear(head_in, 400),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(400),
            nn.Linear(400, 200),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(200),
            nn.Linear(200, 100),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(100),
            nn.Linear(100, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor,
                return_embedding: bool = False,
                return_nodes: bool = False,
                return_pre_block: bool = False) -> torch.Tensor:
        # Accept [B, n_pairs, n_shells] / [B, n_pairs, n_shells, 1] / [B, 1, n_pairs, n_shells]
        if x.dim() == 4:
            if x.shape[-1] == 1:
                x = x.squeeze(-1)
            elif x.shape[1] == 1:
                x = x.squeeze(1)
            else:
                raise ValueError(f'Unexpected 4D input shape {tuple(x.shape)}')
        assert x.shape[1] == self.n_pairs and x.shape[2] == self.n_shells, \
            f'expected [B, {self.n_pairs}, {self.n_shells}], got {tuple(x.shape)}'

        # [B, n_pairs, n_shells] → [B, n_shells, n_pairs]   shell-as-node
        x = x.transpose(1, 2).contiguous()
        # ---- Stage 1 + 1.5 ----
        h = self.node_proj(x)                          # [B, N, hidden]
        h = self.pe(h)

        if return_pre_block:
            return h                                   # [B, n_shells, hidden]

        # ---- Stage 2: GraphTransformerBlock × n_layers ----
        for block in self.blocks:
            h = block(h, self.A_self, self.A_kipf, self.A_raw)   # [B, N, hidden]

        if return_nodes:
            return h                                   # [B, n_shells, hidden]

        # ---- Stage 3: pooling + dense head ----
        h_pooled = self.pool(h)
        if return_embedding:
            return h_pooled
        return self.classifier(h_pooled).view(-1)

    def extra_repr(self) -> str:
        return (f'GraphTransformer(fused) gnn_type={self.gnn_type}, '
                f'schemes={self.schemes}, n_pairs={self.n_pairs}, '
                f'n_shells={self.n_shells}, hidden={self.hidden_dim}, '
                f'n_layers={self.n_layers}, k_neighbors={self.k_neighbors}, '
                f'avg_degree={self._avg_degree:.2f}, n_heads={self.n_heads}, '
                f'use_ffn={self.use_ffn}, ffn_mult={self.ffn_mult}, '
                f'attn_dropout={self.attn_dropout}, attn=GRAPH(chain-k), '
                f'pos_encoding={self.pos_encoding!r}, '
                f'pooling={self.pooling!r}(out={self.pool.out_dim}), head_dropout=NO')

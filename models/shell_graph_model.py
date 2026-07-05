"""Chain-k adjacency helpers for shell-graph models."""
from __future__ import annotations

import torch


def _build_chain_k_adjacency(n_shells: int, k_neighbors: int) -> dict:
    """Build chain-k adjacency matrices used by shell-graph encoders.

    Returns:
        dict with keys:
            'A_raw':   chain-k adjacency without self-loops (GIN)
            'A_self':  chain-k adjacency with self-loops (GAT mask)
            'A_kipf':  symmetric normalized adjacency (GCN)
    """
    A_raw = torch.zeros(n_shells, n_shells, dtype=torch.float32)
    for i in range(n_shells):
        for offset in range(1, k_neighbors + 1):
            if i + offset < n_shells:
                A_raw[i, i + offset] = 1.0
                A_raw[i + offset, i] = 1.0
    A_self = A_raw + torch.eye(n_shells, dtype=torch.float32)
    deg = A_self.sum(dim=1)
    D_inv_sqrt = torch.diag(deg.pow(-0.5))
    A_kipf = D_inv_sqrt @ A_self @ D_inv_sqrt
    return {'A_raw': A_raw, 'A_self': A_self, 'A_kipf': A_kipf}

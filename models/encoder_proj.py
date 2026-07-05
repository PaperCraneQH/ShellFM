"""Encoder-side projection: hidden -> out_dim (linear or small MLP)."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn


def build_encoder_proj(hidden: int, out_dim: int,
                       proj_cfg: Optional[Dict[str, Any]] = None) -> nn.Module:
    """Build ligand/protein projection head.

    proj_cfg.type:
      - linear (default): Linear(hidden, out_dim)
      - mlp: Linear -> ReLU -> Dropout -> Linear
    """
    cfg = proj_cfg or {}
    ptype = str(cfg.get('type', 'linear')).lower()
    if ptype in ('identity', 'none'):
        if hidden != out_dim:
            raise ValueError(
                f'proj.type=identity requires hidden==out_dim, got {hidden}!={out_dim}')
        return nn.Identity()
    if ptype == 'linear':
        return nn.Linear(hidden, out_dim)
    if ptype == 'mlp':
        mid = int(cfg.get('hidden', max(out_dim * 2, 256)))
        drop = float(cfg.get('dropout', 0.1))
        return nn.Sequential(
            nn.Linear(hidden, mid),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(mid, out_dim),
        )
    raise ValueError(f'Unknown proj.type={ptype!r} (allowed: linear, mlp)')

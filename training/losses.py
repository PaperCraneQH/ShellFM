"""Configurable training loss factory.

Integration timeline
---------------------
2026-05-21: Added a ranking auxiliary loss capability for the ESM2-t33_650M +
            bio_mb_large baseline; see notes/RANKING_LOSS_INTEGRATION_2026-05-21.md
            for background. The original `loss_fn = torch.nn.MSELoss()` at
            train_runner.py:790 was replaced by
            `loss_fn = make_loss_fn(cfg['training'].get('loss', {}))`.

Supported loss_type
-------------------
- 'mse_only'        : MSE only (default, fully backward-compatible with legacy yaml behavior)
- 'mse_with_rank'   : MSE + alpha * margin_ranking_loss (pairwise, hinge)

References
----------
- Burges et al. (2005). Learning to rank using gradient descent. ICML.
- Wang et al. (2025). DeepRLI: A multi-objective framework for universal
  protein-ligand interaction prediction. Digital Discovery.

Only native PyTorch operators are used; no new dependencies are introduced.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
import torch.nn as nn


def _pairwise_margin_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    margin: float = 0.5,
    same_y_eps: float = 1e-3,
) -> torch.Tensor:
    """Full-batch pairwise margin ranking loss (hinge form).

    For any pair (i, j) within the batch (i != j, |y_i - y_j| > same_y_eps):
        l_{i,j} = max(0, margin - sign(y_i - y_j) * (pred_i - pred_j))
    Returns the mean over all valid pairs.

    Parameters
    ----------
    pred       : predictions of shape [B, 1] or [B]
    target     : ground truth of shape [B, 1] or [B]
    margin     : hinge margin (default 0.5)
    same_y_eps : exclude pairs with |y_i - y_j| < same_y_eps to guard against label noise

    Returns
    -------
    A scalar tensor; returns 0 if there is no valid pair.
    """
    p = pred.view(-1)
    t = target.view(-1)
    B = p.size(0)
    if B < 2:
        return torch.zeros((), device=p.device, dtype=p.dtype)

    diff_p = p.unsqueeze(0) - p.unsqueeze(1)   # [B, B]
    diff_t = t.unsqueeze(0) - t.unsqueeze(1)
    sign = torch.sign(diff_t)                  # +1 / -1 / 0
    # simultaneously exclude (i==j) (diff_t=0) and |y_i - y_j| < eps
    mask = (diff_t.abs() > same_y_eps).to(p.dtype)
    hinge = torch.relu(margin - sign * diff_p)
    valid = mask.sum().clamp(min=1.0)
    return (hinge * mask).sum() / valid


class _MSEWithRank(nn.Module):
    """nn.Module wrapper that also exposes last_mse / last_rank for training-log monitoring.

    forward returns (mse + rank_weight * rank); during backward the two gradient
    paths are automatically weighted by rank_weight.
    """

    def __init__(self, rank_weight: float, margin: float, same_y_eps: float):
        super().__init__()
        self.rank_weight = float(rank_weight)
        self.margin = float(margin)
        self.same_y_eps = float(same_y_eps)
        self.mse = nn.MSELoss()
        # the training loop can read last_mse / last_rank for logging
        self.last_mse: Optional[float] = None
        self.last_rank: Optional[float] = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l_mse = self.mse(pred, target)
        l_rank = _pairwise_margin_rank_loss(
            pred, target, margin=self.margin, same_y_eps=self.same_y_eps,
        )
        # cache the values for logging (.item() triggers a sync, but it is only a single scalar, so the cost is negligible)
        with torch.no_grad():
            self.last_mse = float(l_mse.detach())
            self.last_rank = float(l_rank.detach())
        return l_mse + self.rank_weight * l_rank


def make_loss_fn(cfg: Optional[Dict] = None) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build a loss function from the yaml configuration.

    yaml field location:
        training:
          loss:
            type: mse_only | mse_with_rank
            rank_weight: 0.10           # only used by mse_with_rank
            margin: 0.5                 # only used by mse_with_rank
            same_y_eps: 1.0e-3          # only used by mse_with_rank

    Parameters
    ----------
    cfg : the training.loss sub-dict; may be None / an empty dict (falls back to mse_only)

    Returns
    -------
    callable(pred, target) -> scalar tensor

    Notes
    -----
    - The returned object additionally carries .last_mse / .last_rank attributes
      (only for mse_with_rank), usable for dual-axis training-log monitoring.
      These two attributes do not exist on the mse_only path.
    """
    cfg = dict(cfg or {})
    loss_type = str(cfg.get('type', 'mse_only')).strip()

    if loss_type == 'mse_only':
        return nn.MSELoss()

    if loss_type == 'mse_with_rank':
        return _MSEWithRank(
            rank_weight=float(cfg.get('rank_weight', 0.10)),
            margin=float(cfg.get('margin', 0.5)),
            same_y_eps=float(cfg.get('same_y_eps', 1.0e-3)),
        )

    raise ValueError(
        f'Unknown training.loss.type={loss_type!r}. '
        f'Valid: mse_only | mse_with_rank'
    )

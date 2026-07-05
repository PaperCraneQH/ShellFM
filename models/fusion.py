"""Fusion heads for ligand + protein (+ optional structure) representations."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(h: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return h.mean(dim=1)
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def _masked_max(h: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is not None:
        neg = torch.finfo(h.dtype).min
        h = h.masked_fill(~mask.bool().unsqueeze(-1), neg)
    return h.max(dim=1).values


def _to_vec(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    return _masked_mean(x, mask) if x.dim() == 3 else x


def _tri_mlp_head(in_dim: int, hidden: int, mid: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, mid),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(mid, 1),
    )


class ConcatMLPFusion(nn.Module):
    def __init__(self, d_lig: int = 128, d_prot: int = 128,
                 hidden: int = 1024, mid: int = 256, dropout: float = 0.2):
        super().__init__()
        self.head = _tri_mlp_head(d_lig + d_prot, hidden, mid, dropout)

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        prot = _to_vec(prot, prot_mask)
        lig = _to_vec(lig, lig_mask)
        return self.head(torch.cat([lig, prot], dim=1))


class BilinearInteractionFusion(nn.Module):
    def __init__(self, d_lig: int = 128, d_prot: int = 128,
                 hidden: int = 1024, mid: int = 256, dropout: float = 0.2):
        super().__init__()
        assert d_lig == d_prot
        self.head = _tri_mlp_head(4 * d_lig, hidden, mid, dropout)

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        prot = _to_vec(prot, prot_mask)
        lig = _to_vec(lig, lig_mask)
        x = torch.cat([lig, prot, lig * prot, (lig - prot).abs()], dim=1)
        return self.head(x)


class CrossAttnGatedFusion(nn.Module):
    def __init__(self, d_lig: int = 128, d_prot: int = 128,
                 n_heads: int = 4, dropout: float = 0.1,
                 use_gate: bool = True):
        super().__init__()
        assert d_lig == d_prot
        d = d_lig
        self.use_gate = bool(use_gate)
        self.attn_lig = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.attn_prot = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        if self.use_gate:
            self.gate = nn.Linear(2 * d, d)
            head_in = d
        else:
            self.gate = None
            head_in = 2 * d
        self.head = nn.Sequential(
            nn.Linear(head_in, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 1)
        )

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert prot.dim() == 3
        lig = _to_vec(lig, lig_mask)
        lig_q = lig.unsqueeze(1)
        key_pad = ~prot_mask.bool() if prot_mask is not None else None
        lig_upd, _ = self.attn_lig(lig_q, prot, prot, key_padding_mask=key_pad)
        if prot_mask is not None:
            m = prot_mask.unsqueeze(-1).float()
            prot_q = (prot * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1.0)
        else:
            prot_q = prot.mean(1, keepdim=True)
        prot_upd, _ = self.attn_prot(prot_q, lig_q, lig_q)
        lig_upd = lig_upd.squeeze(1)
        prot_upd = prot_upd.squeeze(1)
        if self.use_gate:
            g = torch.sigmoid(self.gate(torch.cat([lig_upd, prot_upd], dim=-1)))
            fused = g * lig_upd + (1 - g) * prot_upd
        else:
            fused = torch.cat([lig_upd, prot_upd], dim=-1)
        return self.head(fused)


class LCBCrossAttnFusion(nn.Module):
    def __init__(self, d_lig: int = 128, d_prot: int = 128,
                 n_heads: int = 4, branch_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        assert d_lig == d_prot
        d = d_lig
        self.attn_p = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.attn_l = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.br_p = nn.Linear(d, branch_dim)
        self.br_l = nn.Linear(d, branch_dim)
        self.br_pp = nn.Linear(d, branch_dim)
        self.br_pl = nn.Linear(d, branch_dim)
        self.norm = nn.LayerNorm(4 * branch_dim)
        self.head = nn.Sequential(
            nn.Linear(4 * branch_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert lig.dim() == 3 and prot.dim() == 3
        kp_lig = (~lig_mask.bool()) if lig_mask is not None else None
        kp_prot = (~prot_mask.bool()) if prot_mask is not None else None
        sp_attn, _ = self.attn_p(prot, lig, lig, key_padding_mask=kp_lig)
        sl_attn, _ = self.attn_l(lig, prot, prot, key_padding_mask=kp_prot)
        a_p = F.relu(self.br_p(_masked_max(sp_attn, prot_mask)))
        a_l = F.relu(self.br_l(_masked_max(sl_attn, lig_mask)))
        sp_p = F.relu(self.br_pp(_masked_mean(prot, prot_mask)))
        sp_l = F.relu(self.br_pl(_masked_mean(lig, lig_mask)))
        x = self.norm(torch.cat([a_p, a_l, sp_p, sp_l], dim=1))
        return self.head(x)


class LCBCrossAttnFusion3(nn.Module):
    def __init__(self, d_lig: int = 128, d_prot: int = 128, d_plitext: int = 128,
                 n_heads: int = 4, branch_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        assert d_lig == d_prot
        d = d_lig
        self.attn_p = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.attn_l = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.br_p = nn.Linear(d, branch_dim)
        self.br_l = nn.Linear(d, branch_dim)
        self.br_pp = nn.Linear(d, branch_dim)
        self.br_pl = nn.Linear(d, branch_dim)
        self.br_struct = nn.Linear(d_plitext, branch_dim)
        self.norm = nn.LayerNorm(5 * branch_dim)
        self.head = nn.Sequential(
            nn.Linear(5 * branch_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                plitext: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert lig.dim() == 3 and prot.dim() == 3
        kp_lig = (~lig_mask.bool()) if lig_mask is not None else None
        kp_prot = (~prot_mask.bool()) if prot_mask is not None else None
        sp_attn, _ = self.attn_p(prot, lig, lig, key_padding_mask=kp_lig)
        sl_attn, _ = self.attn_l(lig, prot, prot, key_padding_mask=kp_prot)
        a_p = F.relu(self.br_p(_masked_max(sp_attn, prot_mask)))
        a_l = F.relu(self.br_l(_masked_max(sl_attn, lig_mask)))
        sp_p = F.relu(self.br_pp(_masked_mean(prot, prot_mask)))
        sp_l = F.relu(self.br_pl(_masked_mean(lig, lig_mask)))
        z_s = F.relu(self.br_struct(plitext))
        x = self.norm(torch.cat([a_p, a_l, sp_p, sp_l, z_s], dim=1))
        return self.head(x)


class ConcatMLPFusion3(nn.Module):
    def __init__(self, d_lig: int = 128, d_prot: int = 128, d_plitext: int = 128,
                 hidden: int = 1024, mid: int = 256, dropout: float = 0.2):
        super().__init__()
        self.head = _tri_mlp_head(d_lig + d_prot + d_plitext, hidden, mid, dropout)

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                plitext: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        prot = _to_vec(prot, prot_mask)
        lig = _to_vec(lig, lig_mask)
        return self.head(torch.cat([lig, prot, plitext], dim=1))


class StructOnlyFusion(nn.Module):
    """Structure-only head (component ablation: w/o dual language models).

    Ignores the ligand and protein language-model views and regresses affinity
    from the shell-graph vector alone, isolating the contribution of the PLMs.
    """

    def __init__(self, d_lig: int = 128, d_prot: int = 128, d_plitext: int = 512,
                 hidden: int = 1024, mid: int = 256, dropout: float = 0.2):
        super().__init__()
        self.head = _tri_mlp_head(d_plitext, hidden, mid, dropout)

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                plitext: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.head(plitext)


class BilinearTriFusion(nn.Module):
    """F2: pairwise bilinear interaction + difference features across three modalities -> MLP."""

    def __init__(self, d_lig: int = 128, d_prot: int = 128, d_plitext: int = 128,
                 hidden: int = 1536, mid: int = 640, dropout: float = 0.4):
        super().__init__()
        assert d_lig == d_prot == d_plitext
        d = d_lig
        self.head = _tri_mlp_head(9 * d, hidden, mid, dropout)

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                plitext: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        lig = _to_vec(lig, lig_mask)
        prot = _to_vec(prot, prot_mask)
        s = plitext
        x = torch.cat([
            lig, prot, s,
            lig * prot, lig * s, prot * s,
            (lig - prot).abs(), (lig - s).abs(), (prot - s).abs(),
        ], dim=1)
        return self.head(x)


class GatedTriFusion(nn.Module):
    """F3: learnable 3-way softmax gate then MLP."""

    def __init__(self, d_lig: int = 128, d_prot: int = 128, d_plitext: int = 128,
                 hidden: int = 1536, mid: int = 640, dropout: float = 0.4):
        super().__init__()
        assert d_lig == d_prot == d_plitext
        d = d_lig
        self.gate = nn.Linear(3 * d, 3)
        self.head = _tri_mlp_head(d, hidden, mid, dropout)

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                plitext: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        lig = _to_vec(lig, lig_mask)
        prot = _to_vec(prot, prot_mask)
        s = plitext
        w = F.softmax(self.gate(torch.cat([lig, prot, s], dim=1)), dim=-1)
        fused = w[:, 0:1] * lig + w[:, 1:2] * prot + w[:, 2:3] * s
        return self.head(fused)


class LMXAttnStructFusion(nn.Module):
    """F5: bidirectional LM cross-attn interaction vectors + structure tower concat -> MLP."""

    def __init__(self, d_lig: int = 128, d_prot: int = 128, d_plitext: int = 128,
                 n_heads: int = 4, hidden: int = 1536, mid: int = 640, dropout: float = 0.4):
        super().__init__()
        assert d_lig == d_prot
        d = d_lig
        self.attn_p = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.attn_l = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.head = _tri_mlp_head(2 * d + d_plitext, hidden, mid, dropout)

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                plitext: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert lig.dim() == 3 and prot.dim() == 3
        kp_lig = (~lig_mask.bool()) if lig_mask is not None else None
        kp_prot = (~prot_mask.bool()) if prot_mask is not None else None
        sp_attn, _ = self.attn_p(prot, lig, lig, key_padding_mask=kp_lig)
        sl_attn, _ = self.attn_l(lig, prot, prot, key_padding_mask=kp_prot)
        h_p = _masked_max(sp_attn, prot_mask)
        h_l = _masked_max(sl_attn, lig_mask)
        return self.head(torch.cat([h_p, h_l, plitext], dim=1))


class CosineTriFusion(nn.Module):
    """F6: project three modalities to shared space, mean cosine similarity -> affine pK."""

    def __init__(self, d_lig: int = 128, d_prot: int = 128, d_plitext: int = 128,
                 proj_dim: int = 256, n_proj_layers: int = 2, dropout: float = 0.1,
                 init_scale: float = 3.0, init_bias: float = 6.5):
        super().__init__()
        self.lig_proj = CosineSimilarityFusion._make_proj(d_lig, proj_dim, n_proj_layers, dropout)
        self.prot_proj = CosineSimilarityFusion._make_proj(d_prot, proj_dim, n_proj_layers, dropout)
        self.struct_proj = CosineSimilarityFusion._make_proj(d_plitext, proj_dim, n_proj_layers, dropout)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.bias = nn.Parameter(torch.tensor(float(init_bias)))

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                plitext: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        lig = _to_vec(lig, lig_mask)
        prot = _to_vec(prot, prot_mask)
        z_l = self.lig_proj(lig)
        z_p = self.prot_proj(prot)
        z_s = self.struct_proj(plitext)
        cos_lp = F.cosine_similarity(z_l, z_p, dim=-1)
        cos_ls = F.cosine_similarity(z_l, z_s, dim=-1)
        cos_ps = F.cosine_similarity(z_p, z_s, dim=-1)
        cos_mean = (cos_lp + cos_ls + cos_ps) / 3.0
        return (self.scale * cos_mean + self.bias).unsqueeze(-1)


class HierarchicalFusion(nn.Module):
    """F7: Stage-A bilinear(lig,prot) -> h_lp; Stage-B concat(h_lp, struct) -> MLP。"""

    def __init__(self, d_lig: int = 128, d_prot: int = 128, d_plitext: int = 128,
                 hidden: int = 1536, mid: int = 640, dropout: float = 0.4):
        super().__init__()
        assert d_lig == d_prot
        d = d_lig
        self.lp_proj = nn.Sequential(
            nn.Linear(4 * d, d),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = _tri_mlp_head(d + d_plitext, hidden, mid, dropout)

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                plitext: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        lig = _to_vec(lig, lig_mask)
        prot = _to_vec(prot, prot_mask)
        h_lp = self.lp_proj(torch.cat([lig, prot, lig * prot, (lig - prot).abs()], dim=1))
        return self.head(torch.cat([h_lp, plitext], dim=1))


class CosineSimilarityFusion(nn.Module):
    def __init__(self, d_lig: int = 128, d_prot: int = 128,
                 proj_dim: int = 256, n_proj_layers: int = 2,
                 dropout: float = 0.1,
                 init_scale: float = 3.0, init_bias: float = 6.5):
        super().__init__()
        self.lig_proj = self._make_proj(d_lig, proj_dim, n_proj_layers, dropout)
        self.prot_proj = self._make_proj(d_prot, proj_dim, n_proj_layers, dropout)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.bias = nn.Parameter(torch.tensor(float(init_bias)))

    @staticmethod
    def _make_proj(d_in: int, proj_dim: int, n_layers: int, dropout: float) -> nn.Module:
        layers = []
        d = d_in
        for i in range(max(1, n_layers)):
            last = (i == n_layers - 1)
            layers.append(nn.Linear(d, proj_dim))
            if not last:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            d = proj_dim
        return nn.Sequential(*layers)

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        prot = _to_vec(prot, prot_mask)
        lig = _to_vec(lig, lig_mask)
        z_l = self.lig_proj(lig)
        z_p = self.prot_proj(prot)
        cos = F.cosine_similarity(z_l, z_p, dim=-1)
        return (self.scale * cos + self.bias).unsqueeze(-1)


class WideStructTriFusion(nn.Module):
    """E1: bilinear LM pair (lig/prot) at dim d; structure enters MLP via wide d_plitext.

    Motivation: F02 compresses the structure tower from 7680 to 128 before concat,
    losing discriminative power on structure-dominated splits (CSAR/random/holdout).
    Here we drop equal-dim constraint across towers and give structure a wider
    channel (typical d_plitext=512); LM side still uses [l, p, l*p, |l-p|] low-order interaction.
    """

    def __init__(self, d_lig: int = 128, d_prot: int = 128, d_plitext: int = 512,
                 hidden: int = 1536, mid: int = 640, dropout: float = 0.3):
        super().__init__()
        assert d_lig == d_prot, 'WideStructTriFusion requires d_lig == d_prot'
        self.head = _tri_mlp_head(4 * d_lig + d_plitext, hidden, mid, dropout)

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                plitext: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        lig = _to_vec(lig, lig_mask)
        prot = _to_vec(prot, prot_mask)
        lm = torch.cat([lig, prot, lig * prot, (lig - prot).abs()], dim=1)
        return self.head(torch.cat([lm, plitext], dim=1))


class RawWideConcatFusion(nn.Module):
    """E1-raw: lig/prot keep pretrained pooled dims (e.g. 384/320); wide struct vector; concat -> MLP.

    vs WideStructTriFusion (E1): no LM->128 projection, no [l, p, l⊙p, |l−p|] interaction,
    only concat([h_lig, h_prot, h_struct]).
    """

    def __init__(self, d_lig: int = 384, d_prot: int = 320, d_plitext: int = 512,
                 hidden: int = 1536, mid: int = 640, dropout: float = 0.3):
        super().__init__()
        in_dim = d_lig + d_prot + d_plitext
        self.head = _tri_mlp_head(in_dim, hidden, mid, dropout)

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                plitext: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        lig = _to_vec(lig, lig_mask)
        prot = _to_vec(prot, prot_mask)
        return self.head(torch.cat([lig, prot, plitext], dim=1))


class RawWideXAttnFusion(nn.Module):
    """E1-xattn: native-dim LM (384/320) + wide struct (512) via **residual cross-attention** -> MLP.

    Changes vs RawWideConcatFusion (direct 1216-dim concat)
    --------------------------------------------------------
    1. Three modalities Linear-project to shared d_attn (default 256) + LayerNorm — alignment
       for attention only, not compressing LM semantics to 128.
    2. Stage-A ligand↔protein bidirectional cross-attn (1 token each, residual + LN) — explicit LP complementarity.
    3. Stage-B struct <- LM memory (2 tokens) and each LM branch <- struct (1 token) — wide struct
       channel bidirectionally interacts with LM, avoiding dilution from concat.
    4. Readout concat([h_lig', h_prot', h_struct']) -> 3·d_attn -> MLP (fewer params than raw concat).

    Design notes for small datasets: shallow 2-stage, full residual paths, Dropout, no deep Transformer stack.
    """

    def __init__(self, d_lig: int = 384, d_prot: int = 320, d_plitext: int = 512,
                 d_attn: int = 256, n_heads: int = 4,
                 attn_dropout: float = 0.1, dropout: float = 0.3,
                 hidden: int = 1536, mid: int = 640,
                 skip_lp_xattn: bool = False):
        super().__init__()
        assert d_attn % n_heads == 0, f'd_attn={d_attn} must be divisible by n_heads={n_heads}'
        self.d_attn = int(d_attn)
        self.skip_lp_xattn = bool(skip_lp_xattn)

        self.lig_proj = nn.Linear(d_lig, d_attn)
        self.prot_proj = nn.Linear(d_prot, d_attn)
        self.struct_proj = nn.Linear(d_plitext, d_attn)
        self.ln_lig = nn.LayerNorm(d_attn)
        self.ln_prot = nn.LayerNorm(d_attn)
        self.ln_struct = nn.LayerNorm(d_attn)

        ad = float(attn_dropout)
        if not self.skip_lp_xattn:
            self.lp_l2p = nn.MultiheadAttention(d_attn, n_heads, dropout=ad, batch_first=True)
            self.lp_p2l = nn.MultiheadAttention(d_attn, n_heads, dropout=ad, batch_first=True)
            self.ln_lp_l = nn.LayerNorm(d_attn)
            self.ln_lp_p = nn.LayerNorm(d_attn)
        self.s2lm = nn.MultiheadAttention(d_attn, n_heads, dropout=ad, batch_first=True)
        self.l2s = nn.MultiheadAttention(d_attn, n_heads, dropout=ad, batch_first=True)
        self.p2s = nn.MultiheadAttention(d_attn, n_heads, dropout=ad, batch_first=True)

        self.drop = nn.Dropout(ad)
        self.ln_s2lm = nn.LayerNorm(d_attn)
        self.ln_l2s = nn.LayerNorm(d_attn)
        self.ln_p2s = nn.LayerNorm(d_attn)

        self.head = _tri_mlp_head(3 * d_attn, hidden, mid, dropout)

    @staticmethod
    def _xattn_residual(q: torch.Tensor, mem: torch.Tensor,
                        attn: nn.MultiheadAttention,
                        ln: nn.LayerNorm, drop: nn.Dropout) -> torch.Tensor:
        """q [B,d], mem [B,L,d] → LN(q + Dropout(Attn(q, mem)))."""
        out, _ = attn(q.unsqueeze(1), mem, mem)
        return ln(q + drop(out.squeeze(1)))

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                plitext: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        lig = self.ln_lig(self.lig_proj(_to_vec(lig, lig_mask)))
        prot = self.ln_prot(self.prot_proj(_to_vec(prot, prot_mask)))
        struct = self.ln_struct(self.struct_proj(plitext))

        if self.skip_lp_xattn:
            lig_x, prot_x = lig, prot
        else:
            lig_t = lig.unsqueeze(1)
            prot_t = prot.unsqueeze(1)
            lig_x = self._xattn_residual(lig, prot_t, self.lp_l2p, self.ln_lp_l, self.drop)
            prot_x = self._xattn_residual(prot, lig_t, self.lp_p2l, self.ln_lp_p, self.drop)

        # Stage-B: struct <- [lig, prot]
        lm_mem = torch.stack([lig_x, prot_x], dim=1)
        struct_x = self._xattn_residual(struct, lm_mem, self.s2lm, self.ln_s2lm, self.drop)
        struct_kv = struct_x.unsqueeze(1)

        # Stage-B: lig/prot <- struct (separate queries to avoid shared-KV bias)
        lig_out = self._xattn_residual(lig_x, struct_kv, self.l2s, self.ln_l2s, self.drop)
        prot_out = self._xattn_residual(prot_x, struct_kv, self.p2s, self.ln_p2s, self.drop)

        return self.head(torch.cat([lig_out, prot_out, struct_x], dim=1))


class SymWideXAttnFusion(nn.Module):
    """Symmetric three-modality cross-attention (each modality attends the other two concatenated).

    vs RawWideXAttnFusion (SBCA: all interaction routed through struct z_c)
    -------------------------------------------------------------------------
    SBCA routes all z_l/z_p interaction through struct z_c, hard to describe as
    "standard cross-attention" in papers. This head uses a symmetric, interpretable design:

      Each modality Linear->LN projects to shared width d_attn -> z_l, z_p, z_c;
      each modality uses itself as query and **concat of the other two** as key/value
      (2-token memory) for one multi-head cross-attention step, residual + LN:

        z'_l = LN(z_l + Dropout(MHA(z_l, [z_p; z_c])))
        z'_p = LN(z_p + Dropout(MHA(z_p, [z_l; z_c])))
        z'_c = LN(z_c + Dropout(MHA(z_c, [z_l; z_p])))

      Readout concat([z'_l, z'_p, z'_c]) -> 3·d_attn -> tri-MLP.

    Each modality symmetrically aggregates complementary info from the other two,
    easy to describe as standard all-pairs cross-attention.
    """

    def __init__(self, d_lig: int = 384, d_prot: int = 320, d_plitext: int = 512,
                 d_attn: int = 384, n_heads: int = 6,
                 attn_dropout: float = 0.1, dropout: float = 0.3,
                 hidden: int = 1536, mid: int = 640):
        super().__init__()
        assert d_attn % n_heads == 0, f'd_attn={d_attn} must be divisible by n_heads={n_heads}'
        self.d_attn = int(d_attn)

        self.lig_proj = nn.Linear(d_lig, d_attn)
        self.prot_proj = nn.Linear(d_prot, d_attn)
        self.struct_proj = nn.Linear(d_plitext, d_attn)
        self.ln_lig = nn.LayerNorm(d_attn)
        self.ln_prot = nn.LayerNorm(d_attn)
        self.ln_struct = nn.LayerNorm(d_attn)

        ad = float(attn_dropout)
        self.attn_lig = nn.MultiheadAttention(d_attn, n_heads, dropout=ad, batch_first=True)
        self.attn_prot = nn.MultiheadAttention(d_attn, n_heads, dropout=ad, batch_first=True)
        self.attn_struct = nn.MultiheadAttention(d_attn, n_heads, dropout=ad, batch_first=True)
        self.ln_out_lig = nn.LayerNorm(d_attn)
        self.ln_out_prot = nn.LayerNorm(d_attn)
        self.ln_out_struct = nn.LayerNorm(d_attn)
        self.drop = nn.Dropout(ad)

        self.head = _tri_mlp_head(3 * d_attn, hidden, mid, dropout)

    def _xattn(self, q: torch.Tensor, mem: torch.Tensor,
               attn: nn.MultiheadAttention, ln: nn.LayerNorm) -> torch.Tensor:
        """q [B,d], mem [B,L,d] → LN(q + Dropout(Attn(q, mem)))."""
        out, _ = attn(q.unsqueeze(1), mem, mem)
        return ln(q + self.drop(out.squeeze(1)))

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                plitext: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        z_l = self.ln_lig(self.lig_proj(_to_vec(lig, lig_mask)))
        z_p = self.ln_prot(self.prot_proj(_to_vec(prot, prot_mask)))
        z_c = self.ln_struct(self.struct_proj(plitext))

        # Each modality attends concat of the other two (2-token memory)
        mem_l = torch.stack([z_p, z_c], dim=1)   # lig <- [prot; struct]
        mem_p = torch.stack([z_l, z_c], dim=1)   # prot <- [lig; struct]
        mem_c = torch.stack([z_l, z_p], dim=1)   # struct <- [lig; prot]

        z_l2 = self._xattn(z_l, mem_l, self.attn_lig, self.ln_out_lig)
        z_p2 = self._xattn(z_p, mem_p, self.attn_prot, self.ln_out_prot)
        z_c2 = self._xattn(z_c, mem_c, self.attn_struct, self.ln_out_struct)

        return self.head(torch.cat([z_l2, z_p2, z_c2], dim=1))


class GatedResidualTriFusion(nn.Module):
    """E2/E3/E4: three experts (LM-only / Struct-only / Joint) + per-sample softmax gating.

    Design notes (when fusion underperforms the best single branch on some splits):
      - Three independent heads: y_lm (lig+prot only) / y_st (struct only) / y_fuse (joint);
      - Gate outputs 3-way softmax weights per sample; y = Σ w_k * y_k;
      - Training also returns [y_lm, y_st] for deep supervision so single-modality experts stay strong;
        gate can fall back to y_st on struct-dominated samples or y_lm on LM-dominated ones;
      - When modality_dropout>0, randomly mask LM or Struct expert gates during training
        (prevent over-reliance on one modality); disabled at inference.

    Returns
    -------
    - train(): (y, [y_lm, y_st])  for train_runner deep supervision
    - eval() : y                   single tensor, compatible with other fusion heads
    """

    def __init__(self, d_lig: int = 128, d_prot: int = 128, d_plitext: int = 512,
                 hidden: int = 1536, mid: int = 640, dropout: float = 0.3,
                 modality_dropout: float = 0.0, gate_hidden: int = 128,
                 init_joint_bias: float = 0.0):
        super().__init__()
        assert d_lig == d_prot, 'GatedResidualTriFusion requires d_lig == d_prot'
        self.modality_dropout = float(modality_dropout)
        lm_dim = 4 * d_lig
        # fuse_head is E1 wide joint head (LM interaction 4d + wide struct d_plitext)
        self.lm_head = _tri_mlp_head(lm_dim, hidden, mid, dropout)
        self.st_head = _tri_mlp_head(d_plitext, hidden, mid, dropout)
        self.fuse_head = _tri_mlp_head(lm_dim + d_plitext, hidden, mid, dropout)
        self.gate = nn.Sequential(
            nn.Linear(2 * d_lig + d_plitext, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 3),
        )
        # E5: initial bias on joint expert (index 2) so softmax starts near wide joint head
        # (≈E1); training can shift weight to struct/LM experts (≈E3 fallback) as needed.
        if init_joint_bias != 0.0:
            with torch.no_grad():
                self.gate[-1].bias.copy_(
                    torch.tensor([0.0, 0.0, float(init_joint_bias)]))

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                plitext: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None):
        lig = _to_vec(lig, lig_mask)
        prot = _to_vec(prot, prot_mask)
        lm_feat = torch.cat([lig, prot, lig * prot, (lig - prot).abs()], dim=1)

        y_lm = self.lm_head(lm_feat)
        y_st = self.st_head(plitext)
        y_fuse = self.fuse_head(torch.cat([lm_feat, plitext], dim=1))

        logits = self.gate(torch.cat([lig, prot, plitext], dim=1))   # [B, 3]
        if self.training and self.modality_dropout > 0.0:
            B = logits.size(0)
            r = torch.rand(B, device=logits.device)
            neg = torch.finfo(logits.dtype).min
            drop_lm = r < (self.modality_dropout / 2.0)               # mask LM expert
            drop_st = (r >= (self.modality_dropout / 2.0)) & (r < self.modality_dropout)
            logits = logits.clone()
            logits[drop_lm, 0] = neg
            logits[drop_st, 1] = neg
        w = torch.softmax(logits, dim=-1)                            # [B, 3]
        y = w[:, 0:1] * y_lm + w[:, 1:2] * y_st + w[:, 2:3] * y_fuse

        if self.training:
            return y, [y_lm, y_st]
        return y


class HeteroShellGraphFusion(nn.Module):
    """F08: Hetero-Shell-Graph fusion (LM serves OnionNet-Graph backbone).

    Inputs
    ------
    - lig     : [B, L_lig, d]   per-token ligand features (ChemBERTa per_token, projected to d) + lig_mask
    - prot    : [B, L_prot, d]  per-residue protein features (ESM per_residue, projected to d) + prot_mask
    - plitext : [B, n_shells, d] shell nodes before OnionNet-Graph pooling (return_nodes=True)

    Mechanism (see docs/F08_HeteroShellGraph_LM_Fusion_Design.md §5)
    ----
    Shell nodes as query, ligand tokens / protein residues as KV for cross-attention, injecting LM semantics;
    each path has a **zero-init gate** g=sigmoid(γ), γ₀≪0 -> g≈0 at start, numerically equivalent to pure struct baseline,
    worst case degrades to baseline (theoretically no worse than baseline). Readout flattens 60 shell nodes -> 7680-d, compatible with existing head.

    randomize_lm=True is **negative control NC1**: replace LM features with same-shape random noise (keep mask/attention structure),
    to verify gains come from LM semantics not extra params/attention capacity.
    """

    def __init__(self, d_lig: int = 128, d_prot: int = 128, node_dim: int = 128,
                 n_shells: int = 60, n_layers: int = 1, n_heads: int = 4,
                 dropout: float = 0.2, use_ffn: bool = True, ffn_mult: int = 2,
                 head_hidden: int = 512, head_mid: int = 256,
                 gate_init: float = -4.0, randomize_lm: bool = False,
                 disable_lig: bool = False, disable_prot: bool = False):
        super().__init__()
        assert d_lig == node_dim and d_prot == node_dim, \
            f'HeteroShellGraphFusion requires d_lig==d_prot==node_dim, got {d_lig},{d_prot},{node_dim}'
        if disable_lig and disable_prot:
            raise ValueError('Cannot disable both disable_lig and disable_prot (keep at least one LM injection path)')
        self.n_shells = int(n_shells)
        self.node_dim = int(node_dim)
        self.n_layers = int(n_layers)
        self.randomize_lm = bool(randomize_lm)
        self.use_ffn = bool(use_ffn)
        # A3 ablation: ligand-only / protein-only injection
        self.disable_lig = bool(disable_lig)
        self.disable_prot = bool(disable_prot)

        if not self.disable_lig:
            self.ln_lig_q = nn.ModuleList([nn.LayerNorm(node_dim) for _ in range(n_layers)])
            self.attn_lig = nn.ModuleList([
                nn.MultiheadAttention(node_dim, n_heads, dropout=dropout, batch_first=True)
                for _ in range(n_layers)])
            self.gate_lig = nn.Parameter(torch.full((n_layers,), float(gate_init)))
        if not self.disable_prot:
            self.ln_prot_q = nn.ModuleList([nn.LayerNorm(node_dim) for _ in range(n_layers)])
            self.attn_prot = nn.ModuleList([
                nn.MultiheadAttention(node_dim, n_heads, dropout=dropout, batch_first=True)
                for _ in range(n_layers)])
            self.gate_prot = nn.Parameter(torch.full((n_layers,), float(gate_init)))
        if self.use_ffn:
            self.ffn_ln = nn.ModuleList([nn.LayerNorm(node_dim) for _ in range(n_layers)])
            self.ffn = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(node_dim, ffn_mult * node_dim), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(ffn_mult * node_dim, node_dim),
                ) for _ in range(n_layers)])
            self.gate_ffn = nn.Parameter(torch.full((n_layers,), float(gate_init)))
        self.head = _tri_mlp_head(self.n_shells * node_dim, head_hidden, head_mid, dropout)

    def forward(self, lig: torch.Tensor, prot: torch.Tensor,
                plitext: torch.Tensor,
                prot_mask: Optional[torch.Tensor] = None,
                lig_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert lig.dim() == 3 and prot.dim() == 3 and plitext.dim() == 3, \
            'F08 requires sequence-level lig/prot (per_token/per_residue) and shell nodes (return_nodes=True)'
        h = plitext                                          # [B, n_shells, d] shell nodes
        if self.randomize_lm:                                # NC1 negative control
            lig = torch.randn_like(lig)
            prot = torch.randn_like(prot)
        kp_lig = (~lig_mask.bool()) if lig_mask is not None else None
        kp_prot = (~prot_mask.bool()) if prot_mask is not None else None
        for i in range(self.n_layers):
            if not self.disable_lig:
                a_lig, _ = self.attn_lig[i](self.ln_lig_q[i](h), lig, lig, key_padding_mask=kp_lig)
                h = h + torch.sigmoid(self.gate_lig[i]) * a_lig
            if not self.disable_prot:
                a_prot, _ = self.attn_prot[i](self.ln_prot_q[i](h), prot, prot, key_padding_mask=kp_prot)
                h = h + torch.sigmoid(self.gate_prot[i]) * a_prot
            if self.use_ffn:
                h = h + torch.sigmoid(self.gate_ffn[i]) * self.ffn[i](self.ffn_ln[i](h))
        flat = h.reshape(h.size(0), -1)                      # [B, n_shells*d = 7680]
        return self.head(flat)


def _fusion_common_kwargs(cfg: dict) -> dict:
    return dict(
        hidden=int(cfg.get('hidden', 1024)),
        mid=int(cfg.get('mid', 256)),
        dropout=float(cfg.get('dropout', 0.2)),
    )


def build_fusion(cfg: dict, d_lig: int = 128, d_prot: int = 128,
                 d_plitext: int = 0) -> nn.Module:
    f_type = cfg.get('type', 'concat_mlp')
    kw = _fusion_common_kwargs(cfg)

    if f_type == 'concat_mlp':
        if d_plitext > 0:
            return ConcatMLPFusion3(d_lig=d_lig, d_prot=d_prot, d_plitext=d_plitext, **kw)
        return ConcatMLPFusion(d_lig=d_lig, d_prot=d_prot, **kw)

    if f_type in ('struct_only', 'struct_mlp'):
        assert d_plitext > 0, 'struct_only requires structure tower (plitext.enabled=true)'
        return StructOnlyFusion(d_plitext=d_plitext, **kw)

    if f_type == 'bilinear':
        if d_plitext > 0:
            raise NotImplementedError('Use bilinear_tri for three-branch bilinear fusion.')
        return BilinearInteractionFusion(d_lig=d_lig, d_prot=d_prot, **kw)

    if f_type == 'bilinear_tri':
        assert d_plitext > 0
        return BilinearTriFusion(d_lig=d_lig, d_prot=d_prot, d_plitext=d_plitext, **kw)

    if f_type == 'gated_tri':
        assert d_plitext > 0
        return GatedTriFusion(d_lig=d_lig, d_prot=d_prot, d_plitext=d_plitext, **kw)

    if f_type in ('wide_struct_tri', 'wide_tri'):
        assert d_plitext > 0
        return WideStructTriFusion(d_lig=d_lig, d_prot=d_prot, d_plitext=d_plitext, **kw)

    if f_type in ('raw_wide_concat', 'e1_raw', 'raw_concat_tri'):
        assert d_plitext > 0
        return RawWideConcatFusion(
            d_lig=d_lig, d_prot=d_prot, d_plitext=d_plitext, **kw)

    if f_type in ('raw_wide_xattn', 'e1_rawxattn', 'raw_xattn_tri'):
        assert d_plitext > 0
        return RawWideXAttnFusion(
            d_lig=d_lig, d_prot=d_prot, d_plitext=d_plitext,
            d_attn=int(cfg.get('d_attn', 256)),
            n_heads=int(cfg.get('n_heads', 4)),
            attn_dropout=float(cfg.get('attn_dropout', 0.1)),
            skip_lp_xattn=bool(cfg.get('skip_lp_xattn', False)),
            **kw)

    if f_type in ('sym_wide_xattn', 'sym_xattn_tri', 'symxattn'):
        assert d_plitext > 0
        return SymWideXAttnFusion(
            d_lig=d_lig, d_prot=d_prot, d_plitext=d_plitext,
            d_attn=int(cfg.get('d_attn', 384)),
            n_heads=int(cfg.get('n_heads', 6)),
            attn_dropout=float(cfg.get('attn_dropout', 0.1)),
            **kw)

    if f_type in ('gated_residual_tri', 'gated_res_tri', 'moe_tri'):
        assert d_plitext > 0
        return GatedResidualTriFusion(
            d_lig=d_lig, d_prot=d_prot, d_plitext=d_plitext,
            modality_dropout=float(cfg.get('modality_dropout', 0.0)),
            gate_hidden=int(cfg.get('gate_hidden', 128)),
            init_joint_bias=float(cfg.get('init_joint_bias', 0.0)),
            **kw)

    if f_type in ('hetero_shell_graph', 'f08'):
        assert d_plitext > 0, 'hetero_shell_graph requires struct tower (plitext.enabled=true, return_nodes=true)'
        return HeteroShellGraphFusion(
            d_lig=d_lig, d_prot=d_prot, node_dim=d_plitext,
            n_shells=int(cfg.get('n_shells', 60)),
            n_layers=int(cfg.get('xattn_layers', 1)),
            n_heads=int(cfg.get('n_heads', 4)),
            dropout=float(cfg.get('dropout', 0.2)),
            use_ffn=bool(cfg.get('use_ffn', True)),
            ffn_mult=int(cfg.get('ffn_mult', 2)),
            head_hidden=int(cfg.get('head_hidden', 512)),
            head_mid=int(cfg.get('head_mid', 256)),
            gate_init=float(cfg.get('gate_init', -4.0)),
            randomize_lm=bool(cfg.get('randomize_lm', False)),
            disable_lig=bool(cfg.get('disable_lig', False)),
            disable_prot=bool(cfg.get('disable_prot', False)),
        )

    if f_type == 'lm_xattn_struct':
        assert d_plitext > 0
        return LMXAttnStructFusion(
            d_lig=d_lig, d_prot=d_prot, d_plitext=d_plitext,
            n_heads=int(cfg.get('n_heads', 4)), **kw)

    if f_type == 'cosine_tri':
        assert d_plitext > 0
        return CosineTriFusion(
            d_lig=d_lig, d_prot=d_prot, d_plitext=d_plitext,
            proj_dim=int(cfg.get('proj_dim', 256)),
            n_proj_layers=int(cfg.get('n_proj_layers', 2)),
            dropout=float(cfg.get('dropout', 0.1)),
            init_scale=float(cfg.get('init_scale', 3.0)),
            init_bias=float(cfg.get('init_bias', 6.5)),
        )

    if f_type == 'hierarchical':
        assert d_plitext > 0
        return HierarchicalFusion(d_lig=d_lig, d_prot=d_prot, d_plitext=d_plitext, **kw)

    if f_type == 'cross_attn':
        if d_plitext > 0:
            raise NotImplementedError('cross_attn with structure branch is not supported.')
        return CrossAttnGatedFusion(
            d_lig=d_lig, d_prot=d_prot,
            n_heads=int(cfg.get('n_heads', 4)),
            dropout=float(cfg.get('dropout', 0.1)),
            use_gate=bool(cfg.get('use_gate', True)),
        )

    if f_type in ('lcb_cross_attn', 'lcb'):
        if d_plitext > 0:
            return LCBCrossAttnFusion3(
                d_lig=d_lig, d_prot=d_prot, d_plitext=d_plitext,
                n_heads=int(cfg.get('n_heads', 4)),
                branch_dim=int(cfg.get('branch_dim', 256)),
                dropout=float(cfg.get('dropout', 0.2)),
            )
        return LCBCrossAttnFusion(
            d_lig=d_lig, d_prot=d_prot,
            n_heads=int(cfg.get('n_heads', 4)),
            branch_dim=int(cfg.get('branch_dim', 256)),
            dropout=float(cfg.get('dropout', 0.2)),
        )

    if f_type in ('cosine', 'cosine_sim', 'balm'):
        if d_plitext > 0:
            raise NotImplementedError('Use cosine_tri for three-branch cosine fusion.')
        return CosineSimilarityFusion(
            d_lig=d_lig, d_prot=d_prot,
            proj_dim=int(cfg.get('proj_dim', 256)),
            n_proj_layers=int(cfg.get('n_proj_layers', 2)),
            dropout=float(cfg.get('dropout', 0.1)),
            init_scale=float(cfg.get('init_scale', 3.0)),
            init_bias=float(cfg.get('init_bias', 6.5)),
        )

    raise ValueError(f'Unknown fusion type: {f_type}')

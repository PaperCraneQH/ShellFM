"""Structure tower: residue N60 features -> ShellGraphGTModel -> projection."""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .shell_graph_gt_model import ShellGraphGTModel


class _ResidueFeatStore:
    def __init__(self, features_path: str, scaler_path: Optional[str] = None,
                 missing_policy: str = 'zero'):
        if not os.path.isfile(features_path):
            raise FileNotFoundError(f'_ResidueFeatStore: not found {features_path}')
        z = np.load(features_path, allow_pickle=True)
        codes = z['codes']
        self.features = np.ascontiguousarray(z['feats'].astype(np.float32))
        self.n_features = int(self.features.shape[1])
        self.pdb_to_idx: Dict[str, int] = {
            (str(c).lower().strip()): i for i, c in enumerate(codes)
        }
        self.missing_policy = missing_policy
        self._zero_vec = np.zeros(self.n_features, dtype=np.float32)
        self.n_lookups = 0
        self.n_missing = 0

        if scaler_path and os.path.isfile(scaler_path):
            s = np.load(scaler_path)
            mean = s['mean'].astype(np.float32)
            std = s['std'].astype(np.float32)
        else:
            mean = self.features.mean(axis=0)
            std = self.features.std(axis=0)
            print(f'[ShellGraphStructEncoder][WARN] scaler missing ({scaler_path}); '
                  f'using global stats.', flush=True)
        std = np.where(std < 1e-6, 1.0, std)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def lookup_batch(self, pdb_codes: List[str]) -> np.ndarray:
        out = np.empty((len(pdb_codes), self.n_features), dtype=np.float32)
        for b, pdb in enumerate(pdb_codes):
            self.n_lookups += 1
            key = (pdb or '').lower().strip()
            idx = self.pdb_to_idx.get(key, None)
            if idx is None:
                self.n_missing += 1
                if self.missing_policy == 'zero':
                    out[b] = self._zero_vec
                else:
                    raise KeyError(f'PDB {pdb!r} not in residue_N60 store')
            else:
                out[b] = self.features[idx]
        return out


class ShellGraphStructEncoder(nn.Module):
    """Structure tower backed by the Graph-Transformer shell-graph encoder."""

    TRAIN_MODES = ('e2e', 'freeze_backbone', 'pretrained_init')

    def __init__(self,
                 features_path: str,
                 scaler_path: Optional[str] = None,
                 out_dim: int = 128,
                 n_pairs: int = 168,
                 n_shells: int = 60,
                 gnn_type: str = 'GAT_GCN',
                 hidden_dim: int = 128,
                 n_layers: int = 3,
                 k_neighbors: int = 2,
                 n_heads: int = 4,
                 lstm_num_layers: int = 1,
                 use_ffn: bool = True,
                 pos_encoding: str = 'none',
                 pooling: str = 'flatten',
                 dropout: float = 0.1,
                 proj_dropout: float = 0.2,
                 missing_policy: str = 'zero',
                 train_mode: str = 'e2e',
                 tf_ffn_mult: int = 2,
                 tf_dropout: float = -1.0,
                 **kwargs):
        super().__init__()
        self.n_pairs = int(n_pairs)
        self.n_shells = int(n_shells)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.train_mode = str(train_mode).lower()
        if self.train_mode not in self.TRAIN_MODES:
            raise ValueError(f'train_mode must be one of {self.TRAIN_MODES}, got {train_mode!r}')

        self.store = _ResidueFeatStore(features_path, scaler_path=scaler_path,
                                       missing_policy=missing_policy)
        self.register_buffer('feat_mean', torch.from_numpy(self.store.mean), persistent=False)
        self.register_buffer('feat_std', torch.from_numpy(self.store.std), persistent=False)

        self.backbone = ShellGraphGTModel(
            in_shape=(self.n_pairs, self.n_shells, 1),
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            k_neighbors=k_neighbors,
            gnn_type=gnn_type,
            n_heads=n_heads,
            dropout=dropout,
            lstm_num_layers=lstm_num_layers,
            use_ffn=use_ffn,
            ffn_mult=tf_ffn_mult,
            attn_dropout=tf_dropout,
            pos_encoding=pos_encoding,
            pooling=pooling,
        )
        pool_out = self.backbone.pool.out_dim
        self.backbone.classifier = nn.Identity()
        self.proj = nn.Sequential(
            nn.Linear(pool_out, out_dim),
            nn.ReLU(),
            nn.Dropout(proj_dropout),
        )
        self.apply_train_mode()

    def apply_train_mode(self) -> None:
        if self.train_mode == 'e2e':
            for p in self.backbone.parameters():
                p.requires_grad = True
            for p in self.proj.parameters():
                p.requires_grad = True
        else:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.proj.parameters():
                p.requires_grad = True

    def load_pretrained_backbone(self, ckpt_path: str) -> int:
        if not ckpt_path or not os.path.isfile(ckpt_path):
            print(f'[ShellGraphStructEncoder][WARN] pretrained ckpt not found: {ckpt_path}',
                  flush=True)
            return 0
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        src = ckpt.get('state_dict', ckpt)
        mapped = {}
        for k, v in src.items():
            if k.startswith('classifier.'):
                continue
            mapped[f'backbone.{k}'] = v
        missing, unexpected = self.load_state_dict(mapped, strict=False)
        n_loaded = len(mapped) - len([m for m in missing if m.startswith('backbone.')])
        print(f'[ShellGraphStructEncoder] loaded pretrained backbone from {ckpt_path} '
              f'({n_loaded} tensors)', flush=True)
        if unexpected:
            print(f'  unexpected keys (truncated): {unexpected[:5]}', flush=True)
        self.apply_train_mode()
        return n_loaded

    def forward(self, pdb_codes: List[str],
                device: Optional[torch.device] = None) -> torch.Tensor:
        if device is None:
            device = next(self.proj.parameters()).device
        x = self.store.lookup_batch(pdb_codes)
        x_raw = torch.from_numpy(x).to(device)
        x = (x_raw - self.feat_mean) / self.feat_std
        x = x.view(x.size(0), self.n_pairs, self.n_shells)
        h = self.backbone(x, return_embedding=True)
        return self.proj(h)

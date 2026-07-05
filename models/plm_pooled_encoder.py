"""Base class for frozen pooled PLM encoders (ligand/protein).

When cache is enabled, reads pooled HDF5 vectors (ChemBERTaCache format: key -> emb [hidden])
without loading the backbone; only projection, fusion, and structure tower are trained.
"""
from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Callable, List, Optional

import torch
import torch.nn as nn

from data.chemberta_cache import ChemBERTaCache
from .encoder_proj import build_encoder_proj


def _resolve_dtype(name: str) -> torch.dtype:
    if name in ('bf16', 'bfloat16'):
        return torch.bfloat16
    if name in ('fp16', 'float16', 'half'):
        return torch.float16
    return torch.float32


class FrozenPooledPLMEncoder(nn.Module):
    expects_smiles = False

    def __init__(
        self,
        model_name: str,
        hidden_size: int,
        pool: str = 'mean',
        max_len: int = 512,
        out_dim: int = 128,
        dtype: str = 'bfloat16',
        hf_cache_dir: Optional[str] = None,
        cache_path: Optional[str] = None,
        cache_in_memory: bool = True,
        trust_remote_code: bool = False,
        use_fast_tokenizer: bool = True,
        preprocess: Optional[Callable[[str], str]] = None,
        model_loader: str = 'auto',  # auto | t5_encoder
        proj_cfg: Optional[dict] = None,
        log_prefix: str = 'PLM',
    ):
        super().__init__()
        assert pool in ('mean', 'cls', 'max')
        self.model_name = model_name
        self.pool = pool
        self.max_len = int(max_len)
        self.dtype = _resolve_dtype(dtype)
        self.preprocess = preprocess or (lambda x: x)
        self._log_prefix = log_prefix

        cache_ok = bool(cache_path) and os.path.isfile(cache_path)
        self._cache: Optional[ChemBERTaCache] = None
        if cache_path and not cache_ok:
            print(f'[{log_prefix}][warn] cache not found: {cache_path!r}, '
                  f'fallback to ONLINE forward.')

        if cache_ok:
            self._cache = ChemBERTaCache(cache_path, dtype=torch.float32,
                                         in_memory=bool(cache_in_memory))
            self.hidden = self._cache.hidden
            if self._cache.pool != pool:
                print(f'[{log_prefix}][warn] cache pool={self._cache.pool!r} '
                      f'!= requested {pool!r}; using cached pool.')
                self.pool = self._cache.pool
            self.tokenizer = None
            self.backbone = None
        else:
            if hf_cache_dir:
                os.environ.setdefault('HF_HOME', hf_cache_dir)
            from transformers import AutoModel, AutoTokenizer, T5EncoderModel
            kw = {'cache_dir': hf_cache_dir} if hf_cache_dir else {}
            if trust_remote_code:
                kw['trust_remote_code'] = True
            tok_kw = dict(kw)
            if not use_fast_tokenizer:
                tok_kw['use_fast'] = False
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kw)
            if model_loader == 't5_encoder':
                self.backbone = T5EncoderModel.from_pretrained(model_name, **kw)
                self.hidden = int(self.backbone.config.d_model)
            else:
                self.backbone = AutoModel.from_pretrained(model_name, **kw)
                self.hidden = int(self.backbone.config.hidden_size)
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        if hidden_size and cache_ok and self.hidden != hidden_size:
            print(f'[{log_prefix}][warn] hidden_size yaml={hidden_size} '
                  f'!= cache hidden={self.hidden}')
        self.proj = build_encoder_proj(self.hidden, out_dim, proj_cfg)
        self.register_buffer('_device_ref', torch.zeros(1), persistent=False)

    @property
    def has_trainable_esm(self) -> bool:
        return False

    @property
    def adapter_only(self) -> bool:
        return False

    @staticmethod
    def _pool_hidden(h: torch.Tensor, mask: torch.Tensor, pool: str) -> torch.Tensor:
        if pool == 'cls':
            return h[:, 0]
        if pool == 'max':
            neg = torch.finfo(h.dtype).min
            h_m = h.masked_fill(~mask.bool().unsqueeze(-1), neg)
            return h_m.max(dim=1).values
        m = mask.float().unsqueeze(-1)
        return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

    def _tokenize(self, texts: List[str]):
        assert self.tokenizer is not None
        safe = [self.preprocess(t if isinstance(t, str) and len(t) > 0 else 'A')
                for t in texts]
        return self.tokenizer(
            safe, padding=True, truncation=True,
            max_length=self.max_len, return_tensors='pt')

    def forward(self, texts: List[str]) -> torch.Tensor:
        device = self._device_ref.device
        if self._cache is not None:
            return self.proj(self._cache.lookup_batch(texts, device=device))

        assert self.backbone is not None
        tok = self._tokenize(texts)
        ids = tok['input_ids'].to(device, non_blocking=True)
        mask = tok['attention_mask'].to(device, non_blocking=True)
        amp_ctx = torch.cuda.amp.autocast(
            dtype=self.dtype, enabled=(device.type == 'cuda'))
        with torch.no_grad(), amp_ctx:
            out = self.backbone(input_ids=ids, attention_mask=mask)
            h = out.last_hidden_state if hasattr(out, 'last_hidden_state') else out[0]
        h = h.float()
        pooled = self._pool_hidden(h, mask, self.pool)
        return self.proj(pooled)


class GenericPooledProteinEncoder(FrozenPooledPLMEncoder):
    """Generic frozen pooled protein PLM encoder (Ankh/ProtAlbert/XLNet/Electra/...).

    Reads precomputed pooled HDF5 cache indexed by sequence hash; hidden size is inferred
    from the cache, so any protein PLM is supported without loading the backbone.
    """
    expects_smiles = False


class GenericPooledLigandEncoder(FrozenPooledPLMEncoder):
    """Generic frozen pooled ligand PLM encoder (SELFormer/MolT5/...).

    Same as the protein variant but ``expects_smiles=True``: the dataloader passes raw SMILES
    and the cache is keyed by SMILES hash (input transforms like SELFIES apply at cache build time).
    """
    expects_smiles = True

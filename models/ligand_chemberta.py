"""Ligand encoder: ChemBERTa wrapper (frozen / LoRA + optional pooled HDF5 cache).

Mirrors ``ProteinESMEncoder``: SMILES strings go through pretrained ChemBERTa, then a
projection head aligns ligand vectors with the protein branch.

Modes: frozen, frozen+cache, lora (see protein_esm.py for the same pattern).

Input : list of SMILES strings
Output: Tensor [B, out_dim]
"""
from __future__ import annotations

import os
import re as _re
from contextlib import nullcontext
from typing import List, Optional, Union

import torch
import torch.nn as nn

from data.chemberta_cache import ChemBERTaCache  # noqa: E402  (in-project)
from data.partial_chemberta_cache import PartialChemBERTaCache  # noqa: E402
from .encoder_proj import build_encoder_proj


def _resolve_dtype(name: str) -> torch.dtype:
    if name in ('bf16', 'bfloat16'):
        return torch.bfloat16
    if name in ('fp16', 'float16', 'half'):
        return torch.float16
    return torch.float32


class ChemBERTaLigandEncoder(nn.Module):
    """ChemBERTa ligand encoder.

    Parameters
    ----------
    model_name : str
        HuggingFace model name; default 'DeepChem/ChemBERTa-77M-MLM' (RoBERTa, hidden=384).
    mode : 'frozen' | 'lora'
    pool : 'mean' | 'cls' | 'max' | 'sum' | 'per_token'
        Pooling over ChemBERTa output [B, L, hidden]. ``per_token`` returns
        (projected per-token matrix [B, L, out_dim], attention_mask [B, L])
        for cross-attention fusion (ligand feature matrix Sm_l in CLB paper). This mode
        **does not support pooled cache** (cache stores pooled vectors only) and falls
        back to online forward automatically.
    out_dim : int
        Projected dimension; default 128, aligned with the protein branch.
    cache_path : Optional[str]
        Path to precomputed pooled-vector HDF5. **Frozen + non-per_token only**; when
        set and file exists, backbone is not loaded; forward looks up by SMILES.
    """

    expects_smiles = True   # for DTAModel routing (vs GNN encoder consuming PyG batch)

    def __init__(self,
                 model_name: str = 'DeepChem/ChemBERTa-77M-MLM',
                 mode: str = 'frozen',
                 pool: str = 'mean',
                 max_len: int = 256,
                 out_dim: int = 128,
                 dtype: str = 'bfloat16',
                 hf_cache_dir: Optional[str] = None,
                 gradient_checkpointing: bool = False,
                 lora_r: int = 0,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.05,
                 lora_target_modules: Optional[List[str]] = None,
                 lora_layers: Optional[List[int]] = None,
                 adapter_cfg: Optional[dict] = None,
                 cache_path: Optional[str] = None,
                 cache_in_memory: bool = True,
                 partial_cache_path: Optional[str] = None,
                 per_token_cache_path: Optional[str] = None,
                 proj_cfg: Optional[dict] = None,
                 ):
        super().__init__()
        assert mode in ('frozen', 'lora', 'adapter'), \
            f'mode must be frozen|lora|adapter, got {mode}'
        assert pool in ('mean', 'cls', 'max', 'sum', 'per_token'), \
            f'pool must be mean|cls|max|sum|per_token, got {pool}'
        self.mode = mode
        self.pool = pool
        self.max_len = int(max_len)
        self.dtype = _resolve_dtype(dtype)
        self.model_name = model_name

        # ---- Pooled-vector cache: frozen and non-per_token only ----
        # per_token needs per-token matrix; cache only has pooled vectors -> always online.
        if pool == 'per_token' and cache_path:
            print('[ChemBERTaLigandEncoder][warn] pool=per_token cannot use pooled cache '
                  '(only pooled vectors stored); falling back to online ChemBERTa forward.')
        self._cache: Optional[ChemBERTaCache] = None
        self._partial_cache: Optional[PartialChemBERTaCache] = None
        self._pertoken_cache = None
        self._lora_start_layer: Optional[int] = None
        self._num_layers: Optional[int] = None
        # ---- Per-token frozen cache (for ligand.pool=per_token fusion, e.g. F08-node) ----
        # Frozen ChemBERTa per-token outputs are static; precompute and lookup at train time.
        # Same format as ESM per-residue cache; reuse ESMCache reader.
        pertoken_requested = (bool(per_token_cache_path) and mode == 'frozen'
                              and pool == 'per_token')
        pertoken_available = pertoken_requested and os.path.isfile(per_token_cache_path)  # type: ignore[arg-type]
        if pertoken_requested and not pertoken_available:
            print('[ChemBERTaLigandEncoder][warn] per_token_cache_path='
                  f'{per_token_cache_path!r} not found. Falling back to ONLINE ChemBERTa.\n'
                  '    -> Run first: python scripts/precompute_chemberta_pertoken.py '
                  f'--model {model_name}')
        cache_requested = bool(cache_path) and mode == 'frozen' and pool != 'per_token'
        cache_available = cache_requested and os.path.isfile(cache_path)  # type: ignore[arg-type]
        if cache_requested and not cache_available:
            print(
                '[ChemBERTaLigandEncoder][warn] cache_path='
                f'{cache_path!r} not found. Falling back to ONLINE ChemBERTa '
                'forward.\n'
                '    -> For the fast path, run first:\n'
                '       python scripts/precompute_chemberta_features.py '
                f'--model {model_name}'
            )

        if pertoken_available:
            from data.esm_cache import ESMCache  # same format; reuse (string-keyed [L,H])
            self._pertoken_cache = ESMCache(per_token_cache_path,  # type: ignore[arg-type]
                                            dtype=torch.float32,
                                            in_memory=bool(cache_in_memory))
            self.tokenizer = None
            self.lm = None
            self.hidden = self._pertoken_cache.hidden
            print(f'[ChemBERTaLigandEncoder] per-token cache enabled: '
                  f'{os.path.basename(per_token_cache_path)} (hidden={self.hidden}), '  # type: ignore[arg-type]
                  f'skipping online ChemBERTa forward.')
        elif cache_available:
            self._cache = ChemBERTaCache(cache_path, dtype=torch.float32,  # type: ignore[arg-type]
                                         in_memory=bool(cache_in_memory))
            self.tokenizer = None
            self.lm = None
            self.hidden = self._cache.hidden
            if self._cache.pool != pool:
                print(f'[ChemBERTaLigandEncoder][warn] cache pool='
                      f'{self._cache.pool!r} != requested pool={pool!r}; '
                      f'using cached pooling ({self._cache.pool}).')
                self.pool = self._cache.pool
        else:
            if hf_cache_dir:
                os.environ.setdefault('HF_HOME', hf_cache_dir)
            from transformers import AutoModel, AutoTokenizer  # type: ignore
            kw = {'cache_dir': hf_cache_dir} if hf_cache_dir else {}
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kw)
            self.lm = AutoModel.from_pretrained(model_name, **kw)
            self.hidden = int(self.lm.config.hidden_size)
            self._num_layers = int(self.lm.config.num_hidden_layers)

            # ---- partial cache (LoRA / adapter): skip first N frozen layers ----
            # Must load/validate before PEFT wrap (symmetric with protein_esm.py).
            partial_requested = bool(partial_cache_path) and mode in ('lora', 'adapter')
            partial_available = partial_requested and os.path.isfile(partial_cache_path)  # type: ignore[arg-type]
            if partial_requested and not partial_available:
                raise FileNotFoundError(
                    f'[ChemBERTaLigandEncoder] partial_cache_path='
                    f'{partial_cache_path!r} not found. Run:\n'
                    f'  python scripts/precompute_chemberta_partial_cache.py '
                    f'--model {model_name} --layer <N-1>\n'
                    f'before training, where N is the first trainable layer index.')
            if partial_available:
                self._partial_cache = PartialChemBERTaCache(
                    partial_cache_path,  # type: ignore[arg-type]
                    dtype=torch.float32, in_memory=bool(cache_in_memory))
                # Cache stores layer_idx output -> training runs online from layer_idx+1
                self._lora_start_layer = self._partial_cache.layer_idx + 1
                train_layers: List[int] = []
                if mode == 'lora' and lora_layers:
                    train_layers = [int(l) for l in lora_layers]
                elif mode == 'adapter' and adapter_cfg:
                    train_layers = [int(l) for l in (adapter_cfg.get('layers') or [])]
                if train_layers:
                    bad = [l for l in train_layers
                           if int(l) < self._lora_start_layer]
                    if bad:
                        raise ValueError(
                            f'partial_cache stores layer '
                            f'{self._partial_cache.layer_idx} output, so trainable '
                            f'layers must all be >= {self._lora_start_layer}. '
                            f'Got layers={train_layers}, bad={bad}.')
                print(f'[ChemBERTaLigandEncoder] partial cache enabled: skip '
                      f'layers [0..{self._partial_cache.layer_idx}], train layers '
                      f'[{self._lora_start_layer}..{self._num_layers - 1}] online.')

            if mode == 'frozen':
                for p in self.lm.parameters():
                    p.requires_grad = False
                self.lm.eval()
            elif mode == 'adapter':
                for p in self.lm.parameters():
                    p.requires_grad = False
                if not adapter_cfg or not adapter_cfg.get('enabled', False):
                    raise ValueError('mode=adapter requires ligand.adapter.enabled=true')
                from .petl_inject import inject_adapters
                ad_layers = list(adapter_cfg.get('layers', []) or [])
                if not ad_layers:
                    raise ValueError('mode=adapter requires ligand.adapter.layers')
                inject_adapters(
                    self.lm, ad_layers, self.hidden, adapter_cfg,
                    log_prefix='[ChemBERTaLigandEncoder]',
                )
                self.lm.eval()
            else:  # lora
                from peft import LoraConfig, TaskType, get_peft_model
                for p in self.lm.parameters():
                    p.requires_grad = False
                if gradient_checkpointing:
                    self.lm.gradient_checkpointing_enable()
                    if hasattr(self.lm, 'enable_input_require_grads'):
                        self.lm.enable_input_require_grads()

                # ChemBERTa = RoBERTa: attention submodule paths
                #   encoder.layer.X.attention.self.{query,key,value}
                # Same as ESM/BERT; reuse target_modules / layers regex logic.
                tm_base: List[str] = list(lora_target_modules or ['query', 'value'])
                lora_target: Union[List[str], str]
                n_layers = int(self.lm.config.num_hidden_layers)
                if lora_layers:
                    layer_idxs = sorted({int(i) for i in lora_layers})
                    bad = [i for i in layer_idxs if i < 0 or i >= n_layers]
                    if bad:
                        raise ValueError(
                            f'lora.layers out of range {bad} for '
                            f'num_hidden_layers={n_layers}. '
                            f'Valid range: [0, {n_layers - 1}].')
                    layer_pat = '(?:' + '|'.join(str(i) for i in layer_idxs) + ')'
                    parts: List[str] = []
                    for tm in tm_base:
                        esc = _re.escape(tm)
                        if '.' in tm:
                            parts.append(rf'.*encoder\.layer\.{layer_pat}\.{esc}$')
                        else:
                            parts.append(
                                rf'.*encoder\.layer\.{layer_pat}\.'
                                rf'attention\.self\.{esc}$')
                    lora_target = '(?:' + '|'.join(parts) + ')'
                    self._lora_layers = layer_idxs
                else:
                    lora_target = tm_base
                    self._lora_layers = None

                cfg = LoraConfig(
                    r=int(lora_r),
                    lora_alpha=int(lora_alpha),
                    lora_dropout=float(lora_dropout),
                    bias='none',
                    task_type=TaskType.FEATURE_EXTRACTION,
                    target_modules=lora_target,
                )
                self.lm = get_peft_model(self.lm, cfg)

                if adapter_cfg and adapter_cfg.get('enabled', False):
                    from .petl_inject import inject_adapters
                    ad_layers = list(adapter_cfg.get('layers', []) or [])
                    if not ad_layers:
                        ad_layers = list(self._lora_layers or [])
                    if not ad_layers:
                        raise ValueError(
                            'adapter.enabled=true requires adapter.layers or '
                            'lora.layers to specify target encoder layers.')
                    inject_adapters(
                        self.lm, ad_layers, self.hidden, adapter_cfg,
                        log_prefix='[ChemBERTaLigandEncoder]',
                    )

        self.proj = build_encoder_proj(self.hidden, out_dim, proj_cfg)
        self.register_buffer('_device_ref', torch.zeros(1), persistent=False)

    @staticmethod
    def _pool_sequence(h: torch.Tensor, mask_f: torch.Tensor, pool: str) -> torch.Tensor:
        if pool == 'cls':
            return h[:, 0]
        m = mask_f.unsqueeze(-1)
        if pool == 'sum':
            return (h * m).sum(dim=1)
        if pool == 'max':
            neg = torch.finfo(h.dtype).min
            h_m = h.masked_fill(~mask_f.bool().unsqueeze(-1), neg)
            return h_m.max(dim=1).values
        # mean
        return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

    # ------------------------------------------------------------------
    def _tokenize(self, smiles: List[str]):
        assert self.tokenizer is not None, 'tokenizer disabled (cache mode)'
        # Invalid/empty SMILES fall back to 'C' (methane) to avoid tokenizer errors
        safe = [s if isinstance(s, str) and len(s) > 0 else 'C' for s in smiles]
        return self.tokenizer(
            safe, padding=True, truncation=True,
            max_length=self.max_len, return_tensors='pt',
        )

    @property
    def has_trainable_lm(self) -> bool:
        return self.mode in ('lora', 'adapter')

    @property
    def adapter_only(self) -> bool:
        return self.mode == 'adapter'

    @property
    def uses_cache(self) -> bool:
        return self._cache is not None

    @property
    def uses_partial_cache(self) -> bool:
        return self._partial_cache is not None

    def _get_base_encoder(self):
        """Return the RoBERTa encoder module (works across PEFT wrap)."""
        lm = self.lm
        if hasattr(lm, 'encoder'):
            return lm.encoder
        if hasattr(lm, 'base_model') and hasattr(lm.base_model, 'model'):
            return lm.base_model.model.encoder
        raise RuntimeError('Cannot locate ChemBERTa(RoBERTa) encoder module')

    def _get_extended_attention_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """2D mask [B, L] -> 4D additive mask [B, 1, 1, L] (bidirectional encoder)."""
        try:
            return self.lm.get_extended_attention_mask(mask, mask.shape)
        except Exception:
            ext = mask[:, None, None, :].to(self.dtype)
            return (1.0 - ext) * torch.finfo(self.dtype).min

    # ------------------------------------------------------------------
    def forward(self, smiles: List[str]):
        device = self._device_ref.device

        # ===== Path A0: frozen + per_token cache (for F08-node etc.) =====
        if self._pertoken_cache is not None:
            h, mask = self._pertoken_cache.lookup_batch(
                smiles, max_len=self.max_len, device=device)   # [B, L, hidden], [B, L]
            return self.proj(h), mask

        # ===== Path A: frozen + pooled-vector cache (fastest) =====
        if self._cache is not None:
            pooled = self._cache.lookup_batch(smiles, device=device)  # [B, hidden]
            return self.proj(pooled)

        # ===== Path C: LoRA/adapter + partial cache (first N layers cached) =====
        if self._partial_cache is not None:
            # Cache stores layer_idx output (with <s>/</s>, pad stripped); re-pad per batch
            h_cached, mask = self._partial_cache.lookup_batch(
                smiles, max_len=self.max_len, device=device)
            assert self.lm is not None
            ext_mask = self._get_extended_attention_mask(mask.long())
            amp_ctx = torch.cuda.amp.autocast(dtype=self.dtype,
                                              enabled=(device.type == 'cuda'))
            if self.adapter_only:
                from .petl_inject import set_adapter_modules_train
                set_adapter_modules_train(self._get_base_encoder(), train=True)
            with amp_ctx:
                h = h_cached.to(self.dtype)
                encoder = self._get_base_encoder()
                for layer_idx in range(self._lora_start_layer, self._num_layers):
                    h = encoder.layer[layer_idx](h, attention_mask=ext_mask)[0]
                # RoBERTa encoder has no extra LayerNorm at end; h is last_hidden_state
            h = h.float()
            mask_f = mask.float()
            if self.pool == 'per_token':
                return self.proj(h), mask_f
            if self.pool == 'cls':
                return self.proj(h[:, 0])
            pooled = self._pool_sequence(h, mask_f, self.pool)
            return self.proj(pooled)

        # ===== Path B: online ChemBERTa forward =====
        assert self.lm is not None
        tok = self._tokenize(smiles)
        ids = tok['input_ids'].to(device, non_blocking=True)
        mask = tok['attention_mask'].to(device, non_blocking=True)

        amp_ctx = torch.cuda.amp.autocast(dtype=self.dtype,
                                          enabled=(device.type == 'cuda'))
        grad_ctx = torch.no_grad() if self.mode == 'frozen' else nullcontext()
        if self.adapter_only:
            from .petl_inject import set_adapter_modules_train
            set_adapter_modules_train(self._get_base_encoder(), train=True)

        with grad_ctx, amp_ctx:
            out = self.lm(input_ids=ids, attention_mask=mask)
            h = out.last_hidden_state  # [B, L, H]

        h = h.float()
        mask_f = mask.float()
        if self.pool == 'per_token':
            # Per-token matrix (incl. CLS/EOS) for cross-attention: ([B, L, out_dim], [B, L])
            return self.proj(h), mask_f
        pooled = self._pool_sequence(h, mask_f, self.pool)
        return self.proj(pooled)

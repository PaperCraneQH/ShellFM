"""Protein encoder: ESM-2 wrapper (frozen / LoRA + optional HDF5 cache).

Modes
-----
  - frozen              : freeze ESM-2 backbone; online bf16 forward under no_grad; train projection only
  - frozen + cache      : skip loading ESM-2; read final-layer per-residue features from HDF5 cache
  - lora                : LoRA on attention Q/V with optional gradient checkpointing
  - lora + partial_cache: cache early-layer hidden states; run only upper layers with LoRA at train time

Input : list of protein sequence strings
Output: Tensor [B, out_dim] for pool='mean'/'cls', or per-residue tuple when pool='per_residue'
"""
from __future__ import annotations

import os
from contextlib import nullcontext
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from data.esm_cache import ESMCache  # noqa: E402  (in-project)
from data.partial_esm_cache import PartialESMCache  # noqa: E402
from .encoder_proj import build_encoder_proj


def _resolve_dtype(name: str) -> torch.dtype:
    if name in ('bf16', 'bfloat16'):
        return torch.bfloat16
    if name in ('fp16', 'float16', 'half'):
        return torch.float16
    return torch.float32


class ProteinESMEncoder(nn.Module):
    """ESM-2 protein encoder.

    Parameters
    ----------
    model_name : str
        HuggingFace model name; default 'facebook/esm2_t33_650M_UR50D' (1280-d hidden).
    mode : 'frozen' | 'lora'
        Training mode.
    pool : 'mean' | 'cls' | 'max' | 'sum' | 'per_residue'
        Pooling over ESM output [B, L, hidden]. ``per_residue`` is for cross-attention.
        ``max``/``sum`` aggregate over residue dim after excluding CLS/EOS.
    out_dim : int
        Projected dimension; default 128 to align with the ligand branch.
    cache_path : Optional[str]
        Path to precomputed per-residue HDF5. **Frozen mode only**; when set, the
        backbone is not loaded (saves VRAM and startup time); forward uses cache lookup by seq.
    """

    def __init__(self,
                 model_name: str = 'facebook/esm2_t33_650M_UR50D',
                 mode: str = 'frozen',
                 pool: str = 'mean',
                 max_len: int = 1022,
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
                 proj_cfg: Optional[dict] = None,
                 ):
        super().__init__()
        assert mode in ('frozen', 'lora', 'adapter'), \
            f'mode must be frozen|lora|adapter, got {mode}'
        assert pool in ('mean', 'cls', 'max', 'sum', 'per_residue'), \
            f'pool must be mean|cls|max|sum|per_residue, got {pool}'
        self.mode = mode
        self.pool = pool
        self.max_len = int(max_len)
        self.dtype = _resolve_dtype(dtype)

        # ---- ESM final-layer cache: frozen mode only ----
        self._cache: Optional[ESMCache] = None
        cache_requested = bool(cache_path) and mode == 'frozen'
        cache_available = cache_requested and os.path.isfile(cache_path)  # type: ignore[arg-type]
        if cache_requested and not cache_available:
            print(
                '[ProteinESMEncoder][warn] cache_path='
                f'{cache_path!r} not found. Falling back to ONLINE ESM '
                'forward.\n'
                '    -> For the fast path, run first:\n'
                '       python scripts/precompute_esm_features.py '
                f'--model {model_name}'
            )

        # ---- partial cache (LoRA / adapter): skip first N frozen layers ----
        self._partial_cache: Optional[PartialESMCache] = None
        partial_requested = bool(partial_cache_path) and mode in ('lora', 'adapter')
        partial_available = partial_requested and os.path.isfile(partial_cache_path)  # type: ignore[arg-type]
        if partial_requested and not partial_available:
            raise FileNotFoundError(
                f'[ProteinESMEncoder] partial_cache_path={partial_cache_path!r} '
                f'not found. Run:\n'
                f'  python scripts/precompute_esm_partial_cache.py '
                f'--model {model_name} --layer <N-1>\n'
                f'before training, where N is the first trainable layer index.'
            )

        if cache_available:
            self._cache = ESMCache(cache_path, dtype=torch.float32,  # type: ignore[arg-type]
                                   in_memory=bool(cache_in_memory))
            # With cache, backbone is not loaded at all (tokenizer skipped too)
            self.tokenizer = None
            self.esm = None
            self.hidden = self._cache.hidden
            self._lora_start_layer = None
            if pool == 'cls':
                print('[ProteinESMEncoder][warn] pool=cls is not available when '
                      'using ESM cache (CLS hidden not stored). Falling back to '
                      'pool=mean.')
                self.pool = 'mean'
        else:
            # ---- No cache: load ESM backbone normally ----
            if hf_cache_dir:
                os.environ.setdefault('HF_HOME', hf_cache_dir)

            # Defer import to avoid loading transformers on cache-only path
            from transformers import AutoModel, AutoTokenizer  # type: ignore
            kw = {'cache_dir': hf_cache_dir} if hf_cache_dir else {}
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kw)
            # NOTE: ESM-2 in transformers 4.40 does not support attn_implementation='sdpa'/'flash_attention_2'
            # (raises ValueError), so we use eager here. If a future transformers version adds SDPA,
            # try/except attn_implementation='sdpa' for ~1.2x speedup.
            self.esm = AutoModel.from_pretrained(model_name, **kw)
            self.hidden = self.esm.config.hidden_size

            # ---- Load partial cache (before LoRA wrap, before PEFT renames modules) ----
            self._lora_start_layer: Optional[int] = None
            if partial_available:
                self._partial_cache = PartialESMCache(
                    partial_cache_path,  # type: ignore[arg-type]
                    dtype=torch.float32, in_memory=bool(cache_in_memory),
                )
                # partial cache stores layer_idx output; training starts from layer_idx+1
                self._lora_start_layer = self._partial_cache.layer_idx + 1
                # Validate: all trainable layers must be >= _lora_start_layer
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
                            f'partial_cache stores layer {self._partial_cache.layer_idx} '
                            f'output, so trainable layers must be all >= '
                            f'{self._lora_start_layer}. Got layers={train_layers},'
                            f' bad={bad}.'
                        )
                print(f'[ProteinESMEncoder] partial cache enabled: skip '
                      f'layers [0..{self._partial_cache.layer_idx}], '
                      f'train layers [{self._lora_start_layer}..'
                      f'{self.esm.config.num_hidden_layers - 1}] online.')

            # ---- frozen / lora ----
            if mode == 'frozen':
                for p in self.esm.parameters():
                    p.requires_grad = False
                self.esm.eval()
            elif mode == 'adapter':
                for p in self.esm.parameters():
                    p.requires_grad = False
                if not adapter_cfg or not adapter_cfg.get('enabled', False):
                    raise ValueError('mode=adapter requires protein.adapter.enabled=true')
                from .petl_inject import inject_adapters
                ad_layers = list(adapter_cfg.get('layers', []) or [])
                if not ad_layers:
                    raise ValueError('mode=adapter requires protein.adapter.layers')
                inject_adapters(
                    self.esm, ad_layers, self.hidden, adapter_cfg,
                    log_prefix='[ProteinESMEncoder]',
                )
                self.esm.eval()
            else:  # lora
                import re as _re
                from peft import LoraConfig, TaskType, get_peft_model
                for p in self.esm.parameters():
                    p.requires_grad = False
                if gradient_checkpointing:
                    self.esm.gradient_checkpointing_enable()
                    if hasattr(self.esm, 'enable_input_require_grads'):
                        self.esm.enable_input_require_grads()

                # Default target_modules = ['query', 'value'] -> PEFT matches endswith
                # on all layers' .query / .value.
                # If yaml sets lora_layers (e.g. [4, 5] for last two layers of t6_8M),
                # translate target_modules to regex so PEFT uses re.fullmatch and only
                # attaches to specified submodules on those layers; other layers stay frozen.
                #
                # Target naming (relative to encoder.layer.X):
                #   bare name (no dot): defaults to attention.self.{name}
                #     - 'query' / 'key' / 'value'  -> attention.self.{query,key,value}
                #   dotted path: full path relative to encoder.layer.X
                #     - 'attention.output.dense'   -> attention.output.dense (attn out proj)
                #     - 'intermediate.dense'       -> FFN layer 1
                #     - 'output.dense'             -> FFN layer 2
                # Full 6-module LoRA (NLP standard) recommended:
                #   ['query', 'key', 'value',
                #    'attention.output.dense', 'intermediate.dense', 'output.dense']
                tm_base: List[str] = list(lora_target_modules or ['query', 'value'])
                lora_target: Union[List[str], str]
                if lora_layers:
                    layer_idxs = sorted({int(i) for i in lora_layers})
                    n_layers = int(self.esm.config.num_hidden_layers)
                    bad = [i for i in layer_idxs if i < 0 or i >= n_layers]
                    if bad:
                        raise ValueError(
                            f'lora.layers out of range {bad} for '
                            f'num_hidden_layers={n_layers}. '
                            f'Valid index range: [0, {n_layers - 1}].'
                        )
                    layer_pat = '(?:' + '|'.join(str(i) for i in layer_idxs) + ')'
                    parts: List[str] = []
                    for tm in tm_base:
                        if '.' in tm:
                            # Dotted path: full path, escape and append after encoder.layer.X
                            esc = _re.escape(tm)
                            parts.append(
                                rf'.*encoder\.layer\.{layer_pat}\.{esc}$'
                            )
                        else:
                            # Bare name: default to attention.self.<name>
                            esc = _re.escape(tm)
                            parts.append(
                                rf'.*encoder\.layer\.{layer_pat}\.'
                                rf'attention\.self\.{esc}$'
                            )
                    lora_target = '(?:' + '|'.join(parts) + ')'
                    self._lora_target_regex = lora_target
                    self._lora_layers = layer_idxs
                else:
                    lora_target = tm_base
                    self._lora_target_regex = None
                    self._lora_layers = None

                cfg = LoraConfig(
                    r=int(lora_r),
                    lora_alpha=int(lora_alpha),
                    lora_dropout=float(lora_dropout),
                    bias='none',
                    task_type=TaskType.FEATURE_EXTRACTION,
                    target_modules=lora_target,
                )
                self.esm = get_peft_model(self.esm, cfg)

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
                        self.esm, ad_layers, self.hidden, adapter_cfg,
                        log_prefix='[ProteinESMEncoder]',
                    )

        # ---- projection ----
        self.proj = build_encoder_proj(self.hidden, out_dim, proj_cfg)

    @staticmethod
    def _residue_mask(mask_f: torch.Tensor) -> torch.Tensor:
        """Exclude CLS (col 0) and EOS (last valid) per row."""
        residue_mask = mask_f.clone()
        residue_mask[:, 0] = 0.0
        seq_lens = mask_f.sum(dim=1).long()
        batch_idx = torch.arange(mask_f.shape[0], device=mask_f.device)
        eos_idx = (seq_lens - 1).clamp(min=0)
        residue_mask[batch_idx, eos_idx] = 0.0
        return residue_mask

    def _pool_residue(self, h: torch.Tensor, mask_f: torch.Tensor) -> torch.Tensor:
        if self.pool == 'cls':
            return h[:, 0]
        if self.pool in ('mean', 'max', 'sum'):
            residue_mask = self._residue_mask(mask_f)
            m = residue_mask.unsqueeze(-1)
            if self.pool == 'sum':
                return (h * m).sum(dim=1)
            if self.pool == 'max':
                neg = torch.finfo(h.dtype).min
                h_m = h.masked_fill(~residue_mask.bool().unsqueeze(-1), neg)
                return h_m.max(dim=1).values
            return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        raise ValueError(f'unsupported pool={self.pool!r} for sequence pooling')

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    def _tokenize(self, seqs: List[str]):
        assert self.tokenizer is not None, 'tokenizer disabled (cache mode)'
        safe = [s if isinstance(s, str) and len(s) > 0 else 'A' for s in seqs]
        return self.tokenizer(
            safe,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt',
        )

    @property
    def has_trainable_esm(self) -> bool:
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
        """Return the ESM encoder module (works across PEFT wrap).

        - Unwrapped: self.esm.encoder
        - After PEFT wrap: self.esm.base_model.model.encoder
        PeftModel overrides __getattr__ and forwards unknown attrs to base_model.model,
        so self.esm.encoder usually works. This helper provides a robust fallback.
        """
        esm = self.esm
        if hasattr(esm, 'encoder'):
            return esm.encoder
        if hasattr(esm, 'base_model') and hasattr(esm.base_model, 'model'):
            return esm.base_model.model.encoder
        raise RuntimeError('Cannot locate ESM encoder module')

    def _get_extended_attention_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Expand 2D mask [B, L] to 4D ESM attention mask [B, 1, 1, L].
        Uses HF get_extended_attention_mask; transparent through PEFT wrap.
        """
        return self.esm.get_extended_attention_mask(mask, mask.shape)

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, seqs: List[str]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        device = next(self.parameters()).device

        # ============ Path A: frozen + final-layer HDF5 cache (fastest) ============
        if self._cache is not None:
            # Cache stores actual residue segments (CLS/EOS/pad stripped); mask is all 1
            h, mask = self._cache.lookup_batch(
                seqs, max_len=self.max_len - 2, device=device)
            if self.pool == 'mean':
                m = mask.unsqueeze(-1)
                pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
                return self.proj(pooled)
            h = self.proj(h)
            return h, mask

        # ============ Path C: LoRA/adapter + partial cache (first N layers cached) ============
        if self._partial_cache is not None:
            # Cache stores layer_idx output (with CLS/EOS, padded shape before pad strip)
            # max_len includes CLS/EOS, consistent with tokenizer at train time
            h_cached, mask = self._partial_cache.lookup_batch(
                seqs, max_len=self.max_len, device=device)
            # h_cached: float32 [B, L_total, hidden]
            # mask:     float32 [B, L_total]  (1=valid token incl. CLS/EOS, 0=pad)
            assert self.esm is not None
            # Use long mask for extended_attention_mask
            mask_long = mask.long()
            ext_mask = self._get_extended_attention_mask(mask_long)

            amp_ctx = torch.cuda.amp.autocast(
                dtype=self.dtype, enabled=(device.type == 'cuda'))
            if self.adapter_only:
                from .petl_inject import set_adapter_modules_train
                set_adapter_modules_train(self._get_base_encoder(), train=True)
            with amp_ctx:
                h = h_cached.to(self.dtype)
                encoder = self._get_base_encoder()
                # Run remaining layers layer_idx+1 .. num_layers-1
                for layer_idx in range(self._lora_start_layer,
                                       self.esm.config.num_hidden_layers):
                    layer_out = encoder.layer[layer_idx](
                        h, attention_mask=ext_mask)
                    h = layer_out[0]
                # ESM has an extra emb_layer_norm_after at the end
                if hasattr(encoder, 'emb_layer_norm_after') and \
                        encoder.emb_layer_norm_after is not None:
                    h = encoder.emb_layer_norm_after(h)

            h = h.float()
            mask_f = mask.float()
            if self.pool == 'cls':
                return self.proj(h[:, 0])
            if self.pool in ('mean', 'max', 'sum'):
                pooled = self._pool_residue(h, mask_f)
                return self.proj(pooled)
            # per_residue: return full sequence incl. CLS/EOS; caller can trim if needed
            return self.proj(h), mask_f

        # ============ Path B: online ESM forward (all layers) ============
        assert self.esm is not None
        tok = self._tokenize(seqs)
        ids = tok['input_ids'].to(device, non_blocking=True)
        mask = tok['attention_mask'].to(device, non_blocking=True)

        # Frozen ESM needs no grad: no_grad + bf16;
        # LoRA ESM needs grad (LoRA modules) but can still use autocast bf16.
        amp_ctx = torch.cuda.amp.autocast(dtype=self.dtype, enabled=(device.type == 'cuda'))
        grad_ctx = torch.no_grad() if self.mode == 'frozen' else nullcontext()
        if self.adapter_only:
            from .petl_inject import set_adapter_modules_train
            set_adapter_modules_train(self._get_base_encoder(), train=True)

        with grad_ctx, amp_ctx:
            out = self.esm(input_ids=ids, attention_mask=mask)
            h = out.last_hidden_state  # [B, L, H]

        h = h.float()
        mask_f = mask.float()

        if self.pool == 'cls':
            pooled = h[:, 0]
            return self.proj(pooled)
        if self.pool in ('mean', 'max', 'sum'):
            pooled = self._pool_residue(h, mask_f)
            return self.proj(pooled)
        h = self.proj(h)
        return h, mask_f

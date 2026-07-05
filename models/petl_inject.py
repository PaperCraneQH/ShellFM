"""Inject PEFT adapters into HuggingFace encoder layers.

Supported ``adapter.placement``:
  - parallel        : legacy Step3 — adapter on attention.self, summed into attn output
  - block_parallel  : teacher design — adapter on encoder layer input, summed into layer output
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import torch.nn as nn

from .petl_adapter import AdapterLayer


def _get_encoder_layer_list(model: nn.Module) -> nn.ModuleList:
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        return model.encoder.layer
    base = getattr(model, 'base_model', None)
    if base is not None:
        inner = getattr(base, 'model', base)
        if hasattr(inner, 'encoder') and hasattr(inner.encoder, 'layer'):
            return inner.encoder.layer
    raise RuntimeError('Cannot locate encoder.layer on model (PEFT wrap?)')


def _build_adapter(hidden_size: int, adapter_cfg: Dict[str, Any]) -> AdapterLayer:
    return AdapterLayer(
        d_model=hidden_size,
        bottleneck=int(adapter_cfg.get('bottleneck', 64)),
        dropout=float(adapter_cfg.get('dropout', 0.1)),
        init_option=str(adapter_cfg.get('init_option', 'lora')),
        adapter_scalar=str(adapter_cfg.get('scalar', '1.0')),
        adapter_layernorm_option=str(adapter_cfg.get('layernorm_option', 'in')),
    )


def iter_adapter_modules(encoder: nn.Module) -> Iterable[AdapterLayer]:
    """Yield all injected adapter modules under encoder.layer."""
    for layer in encoder.layer:
        ad = getattr(layer, 'ef_block_adapter', None)
        if ad is not None:
            yield ad
        attn = getattr(layer, 'attention', None)
        if attn is not None:
            ad = getattr(attn.self, 'ef_attn_adapter', None)
            if ad is not None:
                yield ad


def set_adapter_modules_train(encoder: nn.Module, train: bool = True) -> None:
    for ad in iter_adapter_modules(encoder):
        ad.train(train)


def inject_parallel_attn_adapters(
    model: nn.Module,
    layer_indices: Sequence[int],
    hidden_size: int,
    adapter_cfg: Dict[str, Any],
    *,
    log_prefix: str = '',
) -> List[int]:
    """Parallel adapter on attention.self (legacy Step3)."""
    if not adapter_cfg or not adapter_cfg.get('enabled', False):
        return []

    layers = _get_encoder_layer_list(model)
    n_layers = len(layers)
    chosen = sorted({int(i) for i in layer_indices})
    bad = [i for i in chosen if i < 0 or i >= n_layers]
    if bad:
        raise ValueError(
            f'adapter.layers out of range {bad} for num_hidden_layers={n_layers}')

    bottleneck = int(adapter_cfg.get('bottleneck', 64))
    injected: List[int] = []
    for idx in chosen:
        attn_self = layers[idx].attention.self
        if hasattr(attn_self, 'ef_attn_adapter'):
            continue

        adapter = _build_adapter(hidden_size, adapter_cfg)
        attn_self.add_module('ef_attn_adapter', adapter)
        orig_forward = attn_self.forward

        def _wrapped_forward(hidden_states, *args, _orig=orig_forward,
                             _adapter=adapter, **kwargs):
            adapter_out = _adapter(hidden_states, add_residual=False)
            outputs = _orig(hidden_states, *args, **kwargs)
            if isinstance(outputs, tuple):
                context = outputs[0] + adapter_out
                return (context,) + outputs[1:]
            return outputs + adapter_out

        attn_self.forward = _wrapped_forward  # type: ignore[method-assign]
        for p in adapter.parameters():
            p.requires_grad = True
        injected.append(idx)

    if injected and log_prefix:
        print(f'{log_prefix} parallel attn adapters on layers {injected} '
              f'(bottleneck={bottleneck})')
    return injected


def inject_parallel_block_adapters(
    model: nn.Module,
    layer_indices: Sequence[int],
    hidden_size: int,
    adapter_cfg: Dict[str, Any],
    *,
    log_prefix: str = '',
) -> List[int]:
    """Block-parallel adapter: x_{l+1} = EncoderBlock(x_l) + Adapter_l(x_l).

    Each adapter takes the **layer input** hidden states and adds to the full
    encoder layer output (MHA + FFN + residuals), matching UniPELT parallel /
    teacher's per-layer input branch design.
    """
    if not adapter_cfg or not adapter_cfg.get('enabled', False):
        return []

    layers = _get_encoder_layer_list(model)
    n_layers = len(layers)
    chosen = sorted({int(i) for i in layer_indices})
    bad = [i for i in chosen if i < 0 or i >= n_layers]
    if bad:
        raise ValueError(
            f'adapter.layers out of range {bad} for num_hidden_layers={n_layers}')

    bottleneck = int(adapter_cfg.get('bottleneck', 64))
    injected: List[int] = []
    for idx in chosen:
        enc_layer = layers[idx]
        if hasattr(enc_layer, 'ef_block_adapter'):
            continue

        adapter = _build_adapter(hidden_size, adapter_cfg)
        enc_layer.add_module('ef_block_adapter', adapter)
        orig_forward = enc_layer.forward

        def _wrapped_forward(hidden_states, *args, _orig=orig_forward,
                             _adapter=adapter, **kwargs):
            adapter_out = _adapter(hidden_states, add_residual=False)
            outputs = _orig(hidden_states, *args, **kwargs)
            if isinstance(outputs, tuple):
                h = outputs[0] + adapter_out
                return (h,) + outputs[1:]
            return outputs + adapter_out

        enc_layer.forward = _wrapped_forward  # type: ignore[method-assign]
        for p in adapter.parameters():
            p.requires_grad = True
        injected.append(idx)

    if injected and log_prefix:
        print(f'{log_prefix} block-parallel adapters on layers {injected} '
              f'(bottleneck={bottleneck})')
    return injected


def inject_adapters(
    model: nn.Module,
    layer_indices: Sequence[int],
    hidden_size: int,
    adapter_cfg: Dict[str, Any],
    *,
    log_prefix: str = '',
) -> List[int]:
    """Dispatch by ``adapter_cfg.placement``."""
    if not adapter_cfg or not adapter_cfg.get('enabled', False):
        return []
    placement = str(adapter_cfg.get('placement', 'parallel')).lower()
    if placement == 'parallel':
        return inject_parallel_attn_adapters(
            model, layer_indices, hidden_size, adapter_cfg, log_prefix=log_prefix)
    if placement in ('block_parallel', 'block'):
        return inject_parallel_block_adapters(
            model, layer_indices, hidden_size, adapter_cfg, log_prefix=log_prefix)
    raise ValueError(
        f'Unknown adapter.placement={placement!r} '
        f'(allowed: parallel, block_parallel)')


# Backward-compatible alias
inject_parallel_attn_adapters  # noqa: used by external imports

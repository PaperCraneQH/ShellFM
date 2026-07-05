"""Parallel bottleneck Adapter (aligned with unify-parameter-efficient-tuning)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _init_bert_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class AdapterLayer(nn.Module):
    """Bottleneck adapter with optional pre/post LayerNorm and residual."""

    def __init__(
        self,
        d_model: int,
        bottleneck: int,
        dropout: float = 0.0,
        init_option: str = 'lora',
        adapter_scalar: str = '1.0',
        adapter_layernorm_option: str = 'in',
    ):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck
        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option in ('in', 'out'):
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == 'learnable_scalar':
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.dropout = float(dropout)

        if init_option == 'bert':
            self.apply(_init_bert_weights)
        elif init_option == 'lora':
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor, add_residual: bool = True,
                residual: torch.Tensor | None = None) -> torch.Tensor:
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in' and self.adapter_layer_norm_before is not None:
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down) * self.scale

        if self.adapter_layernorm_option == 'out' and self.adapter_layer_norm_before is not None:
            up = self.adapter_layer_norm_before(up)

        return up + residual if add_residual else up

"""ProtT5 protein encoder (frozen + pooled cache, T5 encoder-only)."""
from __future__ import annotations

from typing import Optional

from .plm_pooled_encoder import FrozenPooledPLMEncoder


def _space_join_aa(seq: str) -> str:
    return ' '.join(list(seq))


class ProtT5ProteinEncoder(FrozenPooledPLMEncoder):
    def __init__(
        self,
        model_name: str = 'Rostlab/prot_t5_xl_bfd',
        pool: str = 'mean',
        max_len: int = 512,
        out_dim: int = 1024,
        dtype: str = 'bfloat16',
        hf_cache_dir: Optional[str] = None,
        cache_path: Optional[str] = None,
        cache_in_memory: bool = True,
        proj_cfg: Optional[dict] = None,
    ):
        super().__init__(
            model_name=model_name,
            hidden_size=1024,
            pool=pool,
            max_len=max_len,
            out_dim=out_dim,
            dtype=dtype,
            hf_cache_dir=hf_cache_dir,
            cache_path=cache_path,
            cache_in_memory=cache_in_memory,
            trust_remote_code=False,
            use_fast_tokenizer=False,
            preprocess=_space_join_aa,
            model_loader='t5_encoder',
            proj_cfg=proj_cfg,
            log_prefix='ProtT5ProteinEncoder',
        )

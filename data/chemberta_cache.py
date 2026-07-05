"""HDF5 cache reader for ChemBERTa ligand SMILES pooled features."""
from __future__ import annotations

import hashlib
import os
import threading
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch


def smiles_hash(smiles: str) -> str:
    """First 16 hex chars of sha1(smiles), used as the HDF5 key."""
    return hashlib.sha1(smiles.encode('utf-8')).hexdigest()[:16]


# Module-level in-memory store: { abs_h5_path: (emb [N, H] fp16, {hash: row}) }
_IN_MEM_STORE: Dict[str, Tuple[np.ndarray, Dict[str, int]]] = {}


def _load_h5_to_memory(h5_path: str, hidden: int
                       ) -> Tuple[np.ndarray, Dict[str, int]]:
    import time
    t0 = time.time()
    with h5py.File(h5_path, 'r') as f:
        keys = list(f.keys())
        big = np.empty((len(keys), hidden), dtype=np.float16)
        index: Dict[str, int] = {}
        for row, k in enumerate(keys):
            big[row] = f[k]['emb'][...]
            index[k] = row
    elapsed = time.time() - t0
    mem_mb = big.nbytes / (1024 ** 2)
    print(f'[ChemBERTaCache][in_memory] loaded {len(keys):,} SMILES '
          f'= {mem_mb:.1f} MB fp16  from {os.path.basename(h5_path)}  '
          f'in {elapsed:.1f}s', flush=True)
    return big, index


class ChemBERTaCache:
    """Read-only HDF5 cache: look up pooled features by SMILES string.

    >>> cache = ChemBERTaCache('data_processed_esm/chemberta_cache/chemberta77M_pooled.h5')
    >>> emb = cache.lookup_batch(['CCO', 'c1ccccc1'])   # [B, hidden] float32 Tensor
    """

    def __init__(self, h5_path: str, dtype: torch.dtype = torch.float32,
                 in_memory: bool = True):
        if not os.path.isfile(h5_path):
            raise FileNotFoundError(f'ChemBERTa cache not found: {h5_path}')
        self.h5_path = os.path.abspath(h5_path)
        self._dtype = dtype
        self.in_memory = bool(in_memory)
        with h5py.File(self.h5_path, 'r') as f:
            self.model_name = str(f.attrs.get('model_name', ''))
            self.hidden = int(f.attrs.get('hidden', 0))
            self.pool = str(f.attrs.get('pool', 'mean'))
            self.num_smiles_at_open = int(f.attrs.get('num_smiles', len(f.keys())))

        if self.in_memory:
            if self.h5_path not in _IN_MEM_STORE:
                _IN_MEM_STORE[self.h5_path] = _load_h5_to_memory(
                    self.h5_path, self.hidden)
            self._mem_emb, self._mem_index = _IN_MEM_STORE[self.h5_path]
        self._local = threading.local()

    # ------------------------------------------------------------------
    def _handle(self) -> h5py.File:
        h = getattr(self._local, 'handle', None)
        pid_known = getattr(self._local, 'pid', None)
        cur_pid = os.getpid()
        if h is None or pid_known != cur_pid:
            try:
                h = h5py.File(self.h5_path, 'r', swmr=True)
            except Exception:
                h = h5py.File(self.h5_path, 'r')
            self._local.handle = h
            self._local.pid = cur_pid
        return h

    def __contains__(self, smiles: str) -> bool:
        k = smiles_hash(smiles)
        if self.in_memory:
            return k in self._mem_index
        return k in self._handle()

    def __len__(self) -> int:
        if self.in_memory:
            return len(self._mem_index)
        return len(self._handle().keys())

    def _get_one(self, smiles: str) -> np.ndarray:
        k = smiles_hash(smiles)
        if self.in_memory:
            if k not in self._mem_index:
                raise KeyError(
                    f'SMILES not in ChemBERTa cache (hash={k}, smiles={smiles[:40]!r}). '
                    f'Re-run precompute_chemberta_features.py to refresh.')
            return self._mem_emb[self._mem_index[k]]
        f = self._handle()
        if k not in f:
            raise KeyError(
                f'SMILES not in ChemBERTa cache (hash={k}, smiles={smiles[:40]!r}). '
                f'Re-run precompute_chemberta_features.py to refresh.')
        return f[k]['emb'][...]  # type: ignore[index]

    def lookup_batch(self, smiles: List[str],
                     device: Optional[torch.device] = None) -> torch.Tensor:
        """Look up pooled vectors for a batch of SMILES; return ``[B, hidden]`` Tensor."""
        if not smiles:
            empty = torch.zeros(0, self.hidden, dtype=self._dtype)
            return empty.to(device) if device is not None else empty
        rows = [self._get_one(s) for s in smiles]
        out = np.stack(rows, axis=0).astype(np.float32, copy=False)
        t = torch.from_numpy(out).to(self._dtype)
        if device is not None:
            t = t.to(device, non_blocking=True)
        return t

    def __repr__(self) -> str:
        return (f'ChemBERTaCache(path={os.path.basename(self.h5_path)}, '
                f'model={self.model_name}, hidden={self.hidden}, '
                f'pool={self.pool}, n_smiles={self.num_smiles_at_open})')

    def close(self) -> None:
        h = getattr(self._local, 'handle', None)
        if h is not None:
            try:
                h.close()
            except Exception:
                pass
            self._local.handle = None

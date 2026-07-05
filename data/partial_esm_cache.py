"""HDF5 cache for ESM-2 intermediate hidden states (partial-cache LoRA training)."""
from __future__ import annotations

import hashlib
import os
import threading
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch


def seq_hash(seq: str) -> str:
    return hashlib.sha1(seq.encode('utf-8')).hexdigest()[:16]


_IN_MEM_STORE: Dict[str, Tuple[np.ndarray, Dict[str, Tuple[int, int]]]] = {}


def _load_h5_to_memory(h5_path: str, hidden: int
                       ) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    import time
    t0 = time.time()
    with h5py.File(h5_path, 'r') as f:
        keys = list(f.keys())
        total = 0
        lengths = []
        for k in keys:
            L = int(f[k]['feat'].shape[0])
            lengths.append(L)
            total += L
        big = np.empty((total, hidden), dtype=np.float16)
        index: Dict[str, Tuple[int, int]] = {}
        off = 0
        for k, L in zip(keys, lengths):
            if L == 0:
                index[k] = (off, 0)
                continue
            big[off:off + L] = f[k]['feat'][...]
            index[k] = (off, L)
            off += L
    elapsed = time.time() - t0
    mem_gb = big.nbytes / (1024 ** 3)
    print(f'[PartialESMCache][in_memory] loaded {len(keys):,} seqs / '
          f'{total:,} tokens = {mem_gb:.2f} GB fp16  from '
          f'{os.path.basename(h5_path)}  in {elapsed:.1f}s', flush=True)
    return big, index


class PartialESMCache:
    """Read ESM intermediate hidden states (with CLS/EOS) from HDF5.

    Example
    -------
    >>> cache = PartialESMCache('data_processed_esm/esm_cache/t6_8M_layer3.h5')
    >>> h, mask = cache.lookup_batch(['MKT...', 'GVA...'])
    >>> # h: [B, L_total, hidden]; mask: [B, L_total] (1=valid incl. CLS/EOS)
    >>> # Feed (h, mask) to esm.encoder.layer[layer_idx+1:] during training.
    """

    def __init__(self, h5_path: str, dtype: torch.dtype = torch.float32,
                 in_memory: bool = True):
        if not os.path.isfile(h5_path):
            raise FileNotFoundError(f'PartialESMCache: not found {h5_path}')
        self.h5_path = os.path.abspath(h5_path)
        self._dtype = dtype
        self.in_memory = bool(in_memory)
        with h5py.File(self.h5_path, 'r') as f:
            self.model_name = str(f.attrs.get('model_name', ''))
            self.hidden = int(f.attrs.get('hidden', 0))
            self.layer_idx = int(f.attrs.get('layer_idx', -1))
            self.max_len_input = int(f.attrs.get('max_len_input', 1022))
            self.num_sequences_at_open = int(
                f.attrs.get('num_sequences', len(f.keys())))
        if self.layer_idx < 0:
            raise RuntimeError(f'{self.h5_path}: missing layer_idx attr; '
                               f'corrupted or wrong format.')

        if self.in_memory:
            if self.h5_path not in _IN_MEM_STORE:
                _IN_MEM_STORE[self.h5_path] = _load_h5_to_memory(
                    self.h5_path, self.hidden)
            self._mem_feats, self._mem_index = _IN_MEM_STORE[self.h5_path]
        self._local = threading.local()

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

    def __contains__(self, seq: str) -> bool:
        k = seq_hash(seq)
        if self.in_memory:
            return k in self._mem_index
        return k in self._handle()

    def __len__(self) -> int:
        if self.in_memory:
            return len(self._mem_index)
        return len(self._handle().keys())

    def _get_one(self, seq: str) -> np.ndarray:
        k = seq_hash(seq)
        if self.in_memory:
            if k not in self._mem_index:
                raise KeyError(
                    f'sequence not in partial cache (hash={k}, len={len(seq)}). '
                    f'Re-run precompute_esm_partial_cache.py.')
            off, L = self._mem_index[k]
            return self._mem_feats[off:off + L]
        f = self._handle()
        if k not in f:
            raise KeyError(
                f'sequence not in partial cache (hash={k}, len={len(seq)}). '
                f'Re-run precompute_esm_partial_cache.py.')
        return f[k]['feat'][...]

    def lookup_batch(
        self,
        seqs: List[str],
        max_len: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Look up, pad, and return ``(h, mask)`` for a batch of sequences.

        Parameters
        ----------
        seqs : List[str]
            Raw protein sequence strings.
        max_len : Optional[int]
            Truncation limit (token count including CLS/EOS). None keeps full cached length.
        device : Optional[torch.device]
            Target device.

        Returns
        -------
        h    : Tensor [B, L_max, hidden]  hidden states (with CLS/EOS)
        mask : Tensor [B, L_max]          1=valid token, 0=pad
        """
        if not seqs:
            empty = torch.zeros(0, 0, self.hidden, dtype=self._dtype)
            empty_mask = torch.zeros(0, 0, dtype=self._dtype)
            if device is not None:
                empty = empty.to(device)
                empty_mask = empty_mask.to(device)
            return empty, empty_mask

        feats: List[np.ndarray] = [self._get_one(s) for s in seqs]
        lengths = [a.shape[0] for a in feats]
        if max_len is not None:
            lengths = [min(L, max_len) for L in lengths]
            feats = [a[:L] for a, L in zip(feats, lengths)]
        L_max = max(lengths) if lengths else 0
        B = len(seqs)
        H = self.hidden

        out = np.zeros((B, L_max, H), dtype=np.float32)
        mask = np.zeros((B, L_max), dtype=np.float32)
        for i, (a, L) in enumerate(zip(feats, lengths)):
            if L > 0:
                out[i, :L, :] = a.astype(np.float32, copy=False)
                mask[i, :L] = 1.0

        h = torch.from_numpy(out).to(self._dtype)
        m = torch.from_numpy(mask).to(self._dtype)
        if device is not None:
            h = h.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
        return h, m

    def __repr__(self) -> str:
        extra = ''
        if self.in_memory:
            mem_gb = self._mem_feats.nbytes / (1024 ** 3)
            extra = f', in_memory=True ({mem_gb:.2f}GB)'
        return (f'PartialESMCache(path={os.path.basename(self.h5_path)}, '
                f'model={self.model_name}, layer_idx={self.layer_idx}, '
                f'hidden={self.hidden}, n_seqs={self.num_sequences_at_open}{extra})')

    def close(self) -> None:
        h = getattr(self._local, 'handle', None)
        if h is not None:
            try:
                h.close()
            except Exception:
                pass
            self._local.handle = None

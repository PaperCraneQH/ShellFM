"""BatchSampler that groups samples by similar protein sequence length.

Motivation
----------
PDBbind protein lengths vary widely (mean=551, p99=2412, max=5280). With shuffle=True,
each batch may mix very short and very long sequences. Tokenizers pad to the longest
sequence in the batch, so short sequences waste compute on padding tokens.

LengthBucketSampler sorts samples into K mega-buckets; within each epoch buckets are
shuffled internally and batch order is randomized across buckets, keeping within-batch
lengths similar and reducing padding overhead to ~5% (roughly 1.3–1.5x ESM speedup).

Design notes
------------
1. **Preserves randomness**: bucket order and within-bucket order are re-randomized each epoch.
2. **drop_last defaults to False**: no samples dropped.
3. **Default bucket count** K = max(1, ceil(N / (batch_size * 32))).
4. **Length source**: default ``len(dataset[i].seq)``; optional precomputed ``lengths`` array.

Usage
-----
>>> sampler = LengthBucketSampler(dataset, batch_size=96, num_buckets=50, shuffle=True)
>>> loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=esm_collate_fn,
                        num_workers=4, persistent_workers=True)
"""
from __future__ import annotations

import math
from typing import Iterator, List, Optional, Sequence

import numpy as np
from torch.utils.data import Sampler


class LengthBucketSampler(Sampler[List[int]]):
    """Batch sampler that groups samples of similar protein-sequence length.

    Parameters
    ----------
    dataset : Sequence
        Must expose ``dataset[i].seq`` (str), or pass ``lengths`` explicitly.
    batch_size : int
    num_buckets : Optional[int]
        Default None -> max(1, ceil(N / (batch_size * 32))).
    shuffle : bool
        If True, re-randomize each epoch.
    drop_last : bool
        If True, drop trailing batches smaller than ``batch_size``.
    seed : int
        Base RNG seed; each ``__iter__`` uses ``seed + epoch``.
    lengths : Optional[Sequence[int]]
        If provided, skip reading ``.seq`` from the dataset.
    """

    def __init__(self,
                 dataset,
                 batch_size: int,
                 num_buckets: Optional[int] = None,
                 shuffle: bool = True,
                 drop_last: bool = False,
                 seed: int = 0,
                 lengths: Optional[Sequence[int]] = None):
        if batch_size <= 0:
            raise ValueError(f'batch_size must be > 0, got {batch_size}')
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        if lengths is None:
            lengths = self._infer_lengths_from_dataset(dataset)
        self.lengths = np.asarray(list(lengths), dtype=np.int64)
        self.n_samples = len(self.lengths)
        if self.n_samples == 0:
            raise ValueError('LengthBucketSampler got empty dataset')

        if num_buckets is None:
            est = max(1, math.ceil(self.n_samples / (self.batch_size * 32)))
            num_buckets = est
        self.num_buckets = max(1, int(num_buckets))

        order_by_len = np.argsort(self.lengths, kind='stable')
        self._bucket_indices: List[np.ndarray] = np.array_split(
            order_by_len, self.num_buckets)

        if self.drop_last:
            self._n_batches = sum(len(b) // self.batch_size
                                  for b in self._bucket_indices)
        else:
            self._n_batches = sum(math.ceil(len(b) / self.batch_size)
                                  for b in self._bucket_indices)

    @staticmethod
    def _infer_lengths_from_dataset(dataset) -> List[int]:
        """Read protein sequence length per sample.

        Priority:
          1. ``dataset._data.seq`` (PyG InMemoryDataset stores str fields as lists)
          2. ``dataset[i].seq`` (slower fallback)
        """
        inner = getattr(dataset, '_data', None)
        if inner is not None and hasattr(inner, 'seq'):
            seqs = inner.seq
            if isinstance(seqs, (list, tuple)):
                return [len(s) if isinstance(s, str) else 0 for s in seqs]
        out: List[int] = []
        for i in range(len(dataset)):
            d = dataset[i]
            s = getattr(d, 'seq', '')
            out.append(len(s) if isinstance(s, str) else 0)
        return out

    def __len__(self) -> int:
        return self._n_batches

    def set_epoch(self, epoch: int) -> None:
        """Set epoch explicitly for reproducible RNG control."""
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1

        all_batches: List[np.ndarray] = []
        for bucket in self._bucket_indices:
            idx = bucket.copy()
            if self.shuffle and len(idx) > 1:
                rng.shuffle(idx)
            n = len(idx)
            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                if end > n:
                    if self.drop_last:
                        break
                    end = n
                all_batches.append(idx[start:end])

        if self.shuffle and len(all_batches) > 1:
            order = rng.permutation(len(all_batches))
            all_batches = [all_batches[i] for i in order]

        for b in all_batches:
            yield b.tolist()

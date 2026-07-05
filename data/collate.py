"""Custom DataLoader collate_fn.

PyG's default ``DataLoader`` passes a list of ``Data`` objects to ``Batch.from_data_list``.
Tensor fields are concatenated along the batch dimension; string-field behavior varies
across PyG versions and should not be relied on.

This collate_fn explicitly:
  1. Collects ``seq`` from each ``Data`` into ``List[str]`` (for the protein PLM tokenizer);
  2. Collects ``smiles`` into ``List[str]`` (for ChemBERTa / pooled-cache lookup);
  3. Collects ``pdb_code`` into ``List[str]`` (for the optional structure tower);
  4. Removes those string fields before ``Batch.from_data_list`` so PyG does not try to
     concatenate strings as tensors.

Returns ``(pyg_batch, seq_list, smiles_list, pdb_code_list)``.
All consumers (``train_runner.py``, ``tools/evaluate.py``, ``DTAModel.forward``) use this order.
"""
from __future__ import annotations

from typing import List, Tuple

from torch_geometric.data import Batch, Data


def esm_collate_fn(data_list: List[Data]) -> Tuple[Batch, List[str], List[str], List[str]]:
    seqs: List[str] = []
    smiles: List[str] = []
    pdb_codes: List[str] = []
    cleaned: List[Data] = []
    for d in data_list:
        seq = getattr(d, 'seq', '')
        seqs.append(seq if isinstance(seq, str) else '')
        smi = getattr(d, 'smiles', '')
        smiles.append(smi if isinstance(smi, str) else '')
        pdb = getattr(d, 'pdb_code', '')
        pdb_codes.append(pdb if isinstance(pdb, str) else '')
        # Clone and drop string fields so PyG batching stays tensor-only
        d2 = d.clone()
        for k in ('seq', 'smiles', 'pdb_code'):
            if k in d2:
                try:
                    delattr(d2, k)
                except Exception:
                    try:
                        d2[k] = None
                    except Exception:
                        pass
        cleaned.append(d2)

    batch = Batch.from_data_list(cleaned)
    return batch, seqs, smiles, pdb_codes

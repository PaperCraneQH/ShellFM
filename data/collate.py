"""Custom DataLoader collate_fn."""
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

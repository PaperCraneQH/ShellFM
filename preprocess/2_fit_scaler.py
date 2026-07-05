#!/usr/bin/env python3
"""Preprocessing step 2: fit the per-feature z-score statistics (mean/std) for residue_N60.

To avoid test-set leakage, the statistics are computed **only on the training pool
(PL-2020R1, excluding the external test sets CASF-2016 / CSAR-HiQ)**. The result is saved to
residue_N60_scaler.npz (mean[168*N], std[168*N]) and used by ShellGraphStructEncoder to
standardize features at train/inference time.

Note: this is a "global training-pool" standardization (shared across all splits), a
simplified version of per-fold StandardScaler. Its effect on final performance is minor
(only a per-feature normalization).

Usage:
  python 2_fit_scaler.py \
    --npz ../features_residue/residue_N60.npz \
    --out ../features_residue/residue_N60_scaler.npz \
    --train_index /data/PDBbind/PL-2020R1.csv \
    --exclude /data/PDBbind/CASF-2016.csv /data/PDBbind/CSAR-HiQ.csv
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def _codes_from_csv(path: str) -> set:
    df = pd.read_csv(path)
    col = 'PDB_code' if 'PDB_code' in df.columns else df.columns[0]
    return {str(x).lower().strip() for x in df[col]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--npz', default='../features_residue/residue_N60.npz')
    ap.add_argument('--out', default='../features_residue/residue_N60_scaler.npz')
    ap.add_argument('--train_index', required=True,
                    help='training-pool index CSV (e.g. PL-2020R1.csv); stats use pool samples minus --exclude')
    ap.add_argument('--exclude', nargs='*', default=[],
                    help='external test-set CSVs to remove from the training pool (e.g. CASF-2016.csv CSAR-HiQ.csv)')
    args = ap.parse_args()

    print(f'loading {args.npz} ...', flush=True)
    z = np.load(args.npz, allow_pickle=True)
    codes = np.array([str(c).lower().strip() for c in z['codes']])
    feats = z['feats'].astype(np.float32)
    print(f'  feats: {feats.shape}', flush=True)

    pool = _codes_from_csv(args.train_index)
    test: set = set()
    for p in args.exclude:
        if os.path.isfile(p):
            test |= _codes_from_csv(p)
        else:
            print(f'[warn] exclude csv missing: {p}', flush=True)
    train_codes = pool - test
    print(f'  train pool={len(pool)}, external test={len(test)}, '
          f'train(pool-test)={len(train_codes)}', flush=True)

    mask = np.array([c in train_codes for c in codes])
    sub = feats[mask]
    if sub.shape[0] == 0:
        print('[error] no overlap between the training pool and feature codes; '
              'check whether --train_index / --npz match', flush=True)
        return 1
    print(f'  computing statistics over {sub.shape[0]} training-pool complexes', flush=True)

    mean = sub.mean(axis=0).astype(np.float32)
    std = sub.std(axis=0).astype(np.float32)
    n_const = int((std < 1e-6).sum())
    print(f'  constant features (std<1e-6): {n_const} / {std.shape[0]}', flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, mean=mean, std=std,
                        n_train=sub.shape[0], n_features=feats.shape[1])
    print(f'saved -> {args.out}  ({os.path.getsize(args.out) / 1024:.0f} KB)', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Preprocessing step 3: convert PDBbind split CSVs into PyG `.pt` caches."""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.pdbbind_dataset import PDBbindESMDataset  # noqa: E402


class Paths:
    def __init__(self, args):
        self.pdbbind_root = args.pdbbind_root
        self.processed_root = os.path.abspath(args.processed_root)
        self.cold_start_root = args.cold_start_root or os.path.join(self.pdbbind_root, 'cold_start')
        self.seq_smiles_full = args.seq_smiles_full or \
            os.path.join(self.pdbbind_root, 'PL-2020R1_seq_smiles.csv')
        self.train_full = args.train_full or \
            os.path.join(self.pdbbind_root, 'PL-2020R1_train.csv')
        self.casf_csv = args.casf_csv or os.path.join(self.pdbbind_root, 'CASF-2016.csv')
        self.csar_seq_smiles = args.csar_seq_smiles or \
            os.path.join(self.pdbbind_root, 'CSAR-HiQ_seq_smiles.csv')
        self.csar_dedup_csv = args.csar_dedup_csv or \
            os.path.join(self.pdbbind_root, 'CSAR-HiQ_dedup.csv')


def _load_seq_smiles_full(P: Paths) -> pd.DataFrame:
    df = pd.read_csv(P.seq_smiles_full)
    return df.dropna(subset=['smiles', 'sequence', '-logKd/Ki'])


def _make_dataset(P: Paths, name: str, pdb_codes_csv: str,
                  df_seq_smiles: pd.DataFrame, subdir: Optional[str]):
    sd = subdir if subdir is not None else PDBbindESMDataset.infer_subdir(name)
    new_path = os.path.join(P.processed_root, sd, f'{name}.pt') if sd else \
        os.path.join(P.processed_root, f'{name}.pt')
    if os.path.isfile(new_path):
        print(f'[skip] {new_path} already exists.', flush=True)
        return
    df_s = pd.read_csv(pdb_codes_csv)
    df = df_seq_smiles[df_seq_smiles['PDB_code'].isin(df_s['PDB_code'])].dropna()
    smiles = list(df['smiles'])
    seq = list(df['sequence'])
    pk = list(df['-logKd/Ki'])
    pdb_codes = [str(p).lower() for p in df['PDB_code']]
    print(f'[{name}] {len(df)}/{len(df_s)} rows after join -> subdir={sd or "(root)"}', flush=True)
    PDBbindESMDataset(root=P.processed_root, dataset=name, subdir=sd,
                      xd=smiles, xt=seq, y=pk, xp=pdb_codes)


def build_split(P: Paths, split_type: str, n_repeats: int,
                subsets: List[str] = ('train', 'valid', 'test')):
    df_seq = _load_seq_smiles_full(P)
    for repeat in range(n_repeats):
        for s in subsets:
            name = f'{s}_{split_type}_{repeat}'
            csv = os.path.join(P.cold_start_root, split_type, f'{s}_{split_type}_{repeat}.csv')
            if not os.path.isfile(csv):
                print(f'[warn] missing: {csv}', flush=True)
                continue
            _make_dataset(P, name, csv, df_seq, subdir=split_type)


def build_base(P: Paths, n_repeats: int):
    build_split(P, 'base', n_repeats, subsets=('train', 'valid'))


def build_holdout(P: Paths):
    df_seq = _load_seq_smiles_full(P)
    pairs = [
        ('train_holdout', os.path.join(P.cold_start_root, 'holdout', 'train.csv')),
        ('valid_holdout', os.path.join(P.cold_start_root, 'holdout', 'valid_holdout_2018.csv')),
        ('test_holdout',  os.path.join(P.cold_start_root, 'holdout', 'test_holdout_2019.csv')),
    ]
    for name, csv in pairs:
        if not os.path.isfile(csv):
            print(f'[warn] missing: {csv}', flush=True)
            continue
        _make_dataset(P, name, csv, df_seq, subdir='holdout')


def _csar_eval_codes(P: Paths, df_s_csar: pd.DataFrame) -> set:
    """PDB codes for the leakage-free CSAR-HiQ evaluation set (81 complexes)."""
    n_full = len(df_s_csar)
    if os.path.isfile(P.csar_dedup_csv):
        codes = set(pd.read_csv(P.csar_dedup_csv)['PDB_code'])
        print(f'[CSAR-HiQ] eval codes from {P.csar_dedup_csv}: '
              f'{len(codes)}/{n_full} complexes (no train overlap)', flush=True)
        return codes
    if os.path.isfile(P.train_full):
        train_codes = set(pd.read_csv(P.train_full, usecols=['PDB_code'])['PDB_code'])
        overlap = len(set(df_s_csar['PDB_code']) & train_codes)
        codes = set(df_s_csar['PDB_code']) - train_codes
        print(f'[CSAR-HiQ] dedup vs {P.train_full}: {len(codes)}/{n_full} complexes '
              f'({overlap} removed to avoid train leakage)', flush=True)
        return codes
    print(f'[warn] neither {P.csar_dedup_csv} nor {P.train_full} found; '
          f'CSAR not deduplicated — do not use {n_full} complexes for evaluation', flush=True)
    return set(df_s_csar['PDB_code'])


def build_external(P: Paths):
    df_seq_full = _load_seq_smiles_full(P)
    if os.path.isfile(P.casf_csv):
        _make_dataset(P, 'CASF-2016', P.casf_csv, df_seq_full, subdir='external')
    else:
        print(f'[warn] missing: {P.casf_csv}', flush=True)

    if os.path.isfile(P.csar_seq_smiles):
        df_s_csar = pd.read_csv(P.csar_seq_smiles)
        eval_codes = _csar_eval_codes(P, df_s_csar)
        df = df_s_csar[df_s_csar['PDB_code'].isin(eval_codes)]
        df = df.dropna(subset=['smiles', 'sequence', '-logKd/Ki'])
        new_path = os.path.join(P.processed_root, 'external', 'CSAR-HiQ.pt')
        if os.path.isfile(new_path):
            print(f'[skip] {new_path} already exists.', flush=True)
        else:
            print(f'[CSAR-HiQ] writing {len(df)} eval complexes -> external/CSAR-HiQ.pt', flush=True)
            PDBbindESMDataset(root=P.processed_root, dataset='CSAR-HiQ', subdir='external',
                              xd=list(df['smiles']), xt=list(df['sequence']),
                              y=list(df['-logKd/Ki']),
                              xp=[str(p).lower() for p in df['PDB_code']])
    else:
        print(f'[warn] missing: {P.csar_seq_smiles}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', default='all',
                    choices=['all', 'base', 'random', 'scaffold', 'seq_identity',
                             'holdout', 'external'])
    ap.add_argument('--n_repeats', type=int, default=5)
    ap.add_argument('--pdbbind_root', default=os.path.join(_ROOT, 'data', 'splits'),
                    help='directory with seq_smiles CSVs and cold_start/')
    ap.add_argument('--processed_root', default='data_processed_esm')
    ap.add_argument('--cold_start_root', default=None)
    ap.add_argument('--seq_smiles_full', default=None)
    ap.add_argument('--train_full', default=None)
    ap.add_argument('--casf_csv', default=None)
    ap.add_argument('--csar_seq_smiles', default=None)
    ap.add_argument('--csar_dedup_csv', default=None,
                    help='CSAR eval PDB codes (default: CSAR-HiQ_dedup.csv, 81 complexes)')
    args = ap.parse_args()

    P = Paths(args)
    os.makedirs(P.processed_root, exist_ok=True)
    print(f'PROCESSED_ROOT     = {P.processed_root}', flush=True)
    print(f'COLD_START_ROOT    = {P.cold_start_root}', flush=True)
    print(f'SEQ_SMILES_FULL    = {P.seq_smiles_full}', flush=True)

    if args.split == 'all':
        build_base(P, args.n_repeats)
        for s in ['random', 'scaffold', 'seq_identity']:
            build_split(P, s, args.n_repeats)
        build_holdout(P)
        build_external(P)
    elif args.split == 'base':
        build_base(P, args.n_repeats)
    elif args.split == 'holdout':
        build_holdout(P)
    elif args.split == 'external':
        build_external(P)
    else:
        build_split(P, args.split, args.n_repeats)


if __name__ == '__main__':
    main()

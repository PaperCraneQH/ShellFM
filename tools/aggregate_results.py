"""Aggregate 5-fold result CSVs produced by training into mean +/- std per (split, test set)."""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

METRICS = ('rmse', 'mae', 'sd', 'r')


def _summarize_csv(path: str) -> dict:
    df = pd.read_csv(path)
    df = df[pd.to_numeric(df['repeat'], errors='coerce').notna()]
    df = df[df['repeat'].astype(float) >= 0]
    if df.empty:
        return {}
    out = {'file': os.path.basename(path), 'n_fold': len(df)}
    for m in METRICS:
        if m in df.columns:
            vals = df[m].astype(float).to_numpy()
            out[f'{m}_mean'] = float(vals.mean())
            out[f'{m}_std'] = float(vals.std())
    return out


def _find_csvs(results_dir: str, root: str):
    if results_dir:
        return sorted(glob.glob(os.path.join(results_dir, 'result_*.csv')))
    return sorted(glob.glob(os.path.join(root, '**', 'result_*.csv'), recursive=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--results_dir', help='single results directory')
    g.add_argument('--root', help='results root; recurse over all experiments')
    ap.add_argument('--out_csv', default=None)
    args = ap.parse_args()

    csvs = _find_csvs(args.results_dir, args.root)
    if not csvs:
        print('[warn] no result_*.csv found', flush=True)
        return 1

    records = []
    for p in csvs:
        s = _summarize_csv(p)
        if s:
            s['dir'] = os.path.basename(os.path.dirname(p))
            records.append(s)

    if not records:
        print('[warn] CSVs found but no valid per-fold rows', flush=True)
        return 1

    summary = pd.DataFrame(records)
    cols = ['dir', 'file', 'n_fold'] + [f'{m}_{stat}' for m in METRICS for stat in ('mean', 'std')]
    summary = summary[[c for c in cols if c in summary.columns]]

    with pd.option_context('display.max_rows', None, 'display.width', 200,
                           'display.float_format', lambda x: f'{x:.4f}'):
        print(summary.to_string(index=False), flush=True)

    if args.out_csv:
        summary.to_csv(args.out_csv, index=False, float_format='%.4f')
        print(f'\n[agg] wrote -> {args.out_csv}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Preprocessing step 1: build OnionNet-2 residue-atom shell contact features from raw"""
from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from prepare_inputs import (DataRoots, build_pdbbind_year_map,  # noqa: E402
                            convert_lig, find_protein_ligand)

# ============================================================
#  Stage-2 feature constants and computation (168 pairs / onion binning)
# ============================================================
REC_RES = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 'TRP', 'SER',
           'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 'GLU', 'LYS', 'ARG', 'HIS', 'OTH']
LIG_ELE = ['H', 'C', 'O', 'N', 'P', 'S', 'Hal', 'DU']
HAL_ELE = ['F', 'Cl', 'Br', 'I']
KEYS = ['_'.join(x) for x in itertools.product(REC_RES, LIG_ELE)]  # 168, fixed order
KEY_INDEX = {k: i for i, k in enumerate(KEYS)}
_ELE_PAT = re.compile(r'([A-Za-z]+)\d+[+-]')

_N_SHELLS = 60  # overwritten by main(), read by the multiprocessing worker


def _extract_letter(ele: str) -> str:
    m = _ELE_PAT.match(ele)
    return m.group(1) if m else ele


def _get_res(res: str) -> str:
    return res if res in REC_RES else 'OTH'


def _get_ele(ele: str) -> str:
    if ele in LIG_ELE:
        return ele
    if ele in HAL_ELE:
        return 'Hal'
    return 'DU'


def parse_protein(rec_fpath: str):
    """Return (res_list, all_res_xyz_list[nm]), grouped per residue as in OnionNet-2."""
    with open(rec_fpath) as f:
        lines = [x.strip() for x in f.readlines() if x[:4] in ('ATOM', 'HETA')]
    res_list, all_res_xyz, sym_pool = [], [], []
    num = -1
    temp_xyz = []
    for line in lines:
        ele = _extract_letter(line.split()[-1])
        if ele == 'H':
            continue
        num += 1
        res = _get_res(line[17:20].strip())
        sym = line[17:27].strip()
        x = float(line[30:38].strip()); y = float(line[38:46].strip()); z = float(line[46:54].strip())
        if num == 0:
            res_list.append(res); sym_pool.append(sym); temp_xyz.append([x, y, z])
        elif sym == sym_pool[-1]:
            temp_xyz.append([x, y, z])
        else:
            all_res_xyz.append(np.array(temp_xyz) * 0.1)
            temp_xyz = [[x, y, z]]; sym_pool.append(sym); res_list.append(res)
    if temp_xyz:
        all_res_xyz.append(np.array(temp_xyz) * 0.1)
    return res_list, all_res_xyz


def parse_ligand(lig_fpath: str):
    """Return (lig_ele_list, lig_xyz[nm])."""
    with open(lig_fpath) as f:
        lines = [x.strip() for x in f.readlines() if x[:4] in ('ATOM', 'HETA')]
    ele_list, xyz = [], []
    for line in lines:
        x = float(line[30:38].strip()); y = float(line[38:46].strip()); z = float(line[46:54].strip())
        ele_list.append(_get_ele(_extract_letter(line.split()[-1])))
        xyz.append([x, y, z])
    return ele_list, np.array(xyz) * 0.1


def res_atom_min_dist(res_list, all_res_xyz, lig_ele_list, lig_xyz):
    """Minimum distance for each (residue, ligand_atom) pair; order is (res outer, lig inner)."""
    pairs, dists = [], []
    for res, res_xyz in zip(res_list, all_res_xyz):
        dmin = cdist(lig_xyz, res_xyz, metric='euclidean').min(axis=1)
        for ele, d in zip(lig_ele_list, dmin):
            pairs.append(f'{res}_{ele}')
            dists.append(float(d))
    return pairs, np.array(dists)


def onion_bin(pair_idx, dists, N):
    """Onion binning (0.5 A step); returns a shell-major vector [168*N]."""
    ncutoffs = np.linspace(0.1, 0.05 * (N + 1), N)
    contact = (dists[:, None] <= ncutoffs[None, :]).astype(np.float64)
    onion = np.diff(contact, axis=1, prepend=0.0)
    out = np.empty(168 * N, dtype=np.float32)
    for n in range(N):
        out[n * 168:(n + 1) * 168] = np.bincount(pair_idx, weights=onion[:, n], minlength=168)
    return out


def _worker(line: str):
    parts = line.split()
    if len(parts) < 3:
        return None
    code, rec_fp, lig_fp = parts[0], parts[1], parts[2]
    try:
        res_list, all_res_xyz = parse_protein(rec_fp)
        lig_ele, lig_xyz = parse_ligand(lig_fp)
        if len(res_list) == 0 or len(lig_ele) == 0:
            return (code, None)
        pairs, dists = res_atom_min_dist(res_list, all_res_xyz, lig_ele, lig_xyz)
        pair_idx = np.fromiter((KEY_INDEX[p] for p in pairs), dtype=np.int64, count=len(pairs))
        return (code, onion_bin(pair_idx, dists, _N_SHELLS))
    except Exception as e:  # noqa: BLE001
        return (code, f'ERR:{e}')


def _init_pool(n_shells: int):
    global _N_SHELLS
    _N_SHELLS = n_shells


# ============================================================
#  Stage 1: collect codes -> locate -> convert ligand -> inputs.dat
# ============================================================
def _collect_codes(index_csvs, codes_file) -> list:
    if codes_file and os.path.isfile(codes_file):
        with open(codes_file) as f:
            return sorted({x.strip().lower() for x in f if x.strip()})
    codes: set = set()
    for csv in index_csvs:
        if not os.path.isfile(csv):
            print(f'[warn] index csv missing: {csv}', flush=True)
            continue
        df = pd.read_csv(csv)
        col = 'PDB_code' if 'PDB_code' in df.columns else df.columns[0]
        codes.update(df[col].astype(str).str.lower().str.strip().tolist())
    return sorted(codes)


def stage1_prepare(codes, roots: DataRoots, staging: Path) -> Path:
    pdb_dir = staging / 'pdb'
    pdb_dir.mkdir(parents=True, exist_ok=True)
    year_map = build_pdbbind_year_map(roots.pdbbind_index)
    print(f'[stage1] year_map entries = {len(year_map)}', flush=True)

    inputs_lines, missing, convfail = [], [], []
    tool_counts = {'cached': 0, 'rdkit': 0, 'obabel': 0}
    for i, pdb in enumerate(codes):
        if i % 500 == 0:
            print(f'[stage1] {i}/{len(codes)}  ok={len(inputs_lines)} '
                  f'missing={len(missing)} convfail={len(convfail)}', flush=True)
        prot, lig, _src = find_protein_ligand(pdb, year_map, roots)
        if prot is None:
            missing.append(pdb)
            continue
        cur = pdb_dir / pdb
        cur.mkdir(exist_ok=True)
        prot_dst = cur / f'{pdb}_protein.pdb'
        if not prot_dst.exists():
            try:
                os.symlink(prot, prot_dst)
            except FileExistsError:
                pass
        lig_dst = cur / f'{pdb}_ligand.pdb'
        if lig_dst.exists() and lig_dst.stat().st_size > 0:
            ok, tool = True, 'cached'
        else:
            ok, tool = convert_lig(lig, lig_dst)
        if not ok:
            convfail.append(pdb)
            continue
        tool_counts[tool] = tool_counts.get(tool, 0) + 1
        inputs_lines.append(f'{pdb}\t{prot_dst}\t{lig_dst}')

    inputs_path = staging / 'inputs.dat'
    inputs_path.write_text('\n'.join(inputs_lines) + '\n')
    (staging / '_missing.txt').write_text('\n'.join(missing) + '\n')
    (staging / '_convfail.txt').write_text('\n'.join(convfail) + '\n')
    print(f'[stage1][done] prepared={len(inputs_lines)} missing={len(missing)} '
          f'convfail={len(convfail)} tools={tool_counts}', flush=True)
    return inputs_path


def stage2_features(inputs_path: Path, out_dir: Path, n_shells: int, nproc: int):
    with open(inputs_path) as f:
        lines = [x.strip() for x in f if x.strip()]
    print(f'[stage2] complexes={len(lines)}  N={n_shells}  nproc={nproc}', flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    codes_ok, feats, genfail, done = [], [], [], 0
    with mp.Pool(nproc, initializer=_init_pool, initargs=(n_shells,)) as pool:
        for res in pool.imap_unordered(_worker, lines, chunksize=8):
            done += 1
            if done % 1000 == 0:
                print(f'[stage2] {done}/{len(lines)}  ok={len(codes_ok)} fail={len(genfail)}',
                      flush=True)
            if res is None:
                continue
            code, payload = res
            if isinstance(payload, np.ndarray):
                codes_ok.append(code)
                feats.append(payload)
            else:
                genfail.append(f'{code}\t{payload}')

    feats_arr = np.stack(feats, axis=0).astype(np.float32)
    out_npz = out_dir / f'residue_N{n_shells}.npz'
    np.savez_compressed(out_npz, codes=np.array(codes_ok), feats=feats_arr)
    (out_dir / '_genfail.txt').write_text('\n'.join(genfail) + '\n')
    print(f'[stage2][done] saved {out_npz}  shape={feats_arr.shape}  '
          f'ok={len(codes_ok)} fail={len(genfail)}', flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--index_csvs', nargs='+', default=None,
                    help='one or more CSVs (with a PDB_code column) to collect the full set of codes')
    ap.add_argument('--codes_file', default=None,
                    help='optional: a precomputed code list (one per line); overrides --index_csvs')
    ap.add_argument('--casf_root', default=os.environ.get('CASF_ROOT'))
    ap.add_argument('--pdbbind_pl_root', default=os.environ.get('PDBBIND_PL_ROOT'))
    ap.add_argument('--csar_root', default=os.environ.get('CSAR_ROOT'))
    ap.add_argument('--pdbbind_index', default=os.environ.get('PDBBIND_INDEX'),
                    help='PDBbind index CSV (with PDB_code and file/year columns)')
    ap.add_argument('--out_dir', default='../features_residue')
    ap.add_argument('--staging', default='../features_residue/staging')
    ap.add_argument('--n_shells', type=int, default=60)
    ap.add_argument('--nproc', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument('--limit', type=int, default=-1)
    ap.add_argument('--skip_stage1', action='store_true',
                    help='skip locate+convert and compute features directly if staging/inputs.dat exists')
    args = ap.parse_args()

    staging = Path(args.staging)
    out_dir = Path(args.out_dir)
    roots = DataRoots(casf_root=args.casf_root, pdbbind_pl_root=args.pdbbind_pl_root,
                      csar_root=args.csar_root, pdbbind_index=args.pdbbind_index)

    inputs_path = staging / 'inputs.dat'
    if not args.skip_stage1:
        codes = _collect_codes(args.index_csvs or [], args.codes_file)
        if args.limit > 0:
            codes = codes[:args.limit]
        if not codes:
            print('[error] no PDB_code collected; please provide --index_csvs or --codes_file', flush=True)
            return 1
        print(f'[stage1] total unique codes = {len(codes)}', flush=True)
        inputs_path = stage1_prepare(codes, roots, staging)
    elif not inputs_path.is_file():
        print(f'[error] --skip_stage1 set but {inputs_path} not found', flush=True)
        return 1

    stage2_features(inputs_path, out_dir, args.n_shells, args.nproc)
    return 0


if __name__ == '__main__':
    sys.exit(main())

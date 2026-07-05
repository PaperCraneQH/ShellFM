"""Load a trained best checkpoint and evaluate on specified datasets (RMSE / MAE / SD / Pearson r)."""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data import PDBbindESMDataset, esm_collate_fn  # noqa: E402
from models import build_model, setup_struct_branch  # noqa: E402
from utils import Evaluate  # noqa: E402


@torch.no_grad()
def _predict(model, device, loader):
    model.eval()
    preds, labels = [], []
    for pyg_batch, seqs, smiles, pdb_codes in loader:
        pyg_batch = pyg_batch.to(device, non_blocking=True)
        out = model(pyg_batch, seqs, smiles, pdb_codes)
        if isinstance(out, (tuple, list)):
            out = out[0]
        preds.append(out.detach().float().view(-1).cpu())
        labels.append(pyg_batch.y.view(-1).detach().float().cpu())
    if not preds:
        return np.array([]), np.array([])
    return torch.cat(labels).numpy(), torch.cat(preds).numpy()


def _build(cfg: dict, model_name: str, device, split: str, repeat: int):
    model = build_model(cfg, ligand_model_name=model_name).to(device)
    setup_struct_branch(model, cfg.get('plitext') or {}, split=split, repeat=repeat)
    return model


def _load_ckpt(model, ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get('state_dict') if isinstance(ckpt, dict) else ckpt
    if sd is None:
        sd = ckpt.get('latest_state_dict')
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return len(missing), len(unexpected)


def _eval_one(model, device, data_root, name, batch_eval, num_workers):
    ds = PDBbindESMDataset(root=data_root, dataset=name)
    loader = DataLoader(ds, batch_size=batch_eval, shuffle=False,
                        num_workers=num_workers, collate_fn=esm_collate_fn)
    labels, preds = _predict(model, device, loader)
    rmse, mae, r, sd = Evaluate().evaluate(labels, preds)
    return {'name': name, 'n': len(labels), 'rmse': rmse, 'mae': mae, 'sd': sd, 'r': r}


def _discover_ckpts(results_dir: str, model_name: str) -> List[str]:
    pats = sorted(glob.glob(os.path.join(results_dir, f'{model_name}_*_*.pt')))
    return [p for p in pats if os.path.basename(p)[:-3].rsplit('_', 1)[-1].isdigit()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--data_root', default='data_processed_esm')
    ap.add_argument('--test_datasets', nargs='+', required=True,
                    help='dataset .pt names to evaluate, e.g. CASF-2016 CSAR-HiQ test_random_0')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--ckpt', help='single checkpoint .pt path')
    g.add_argument('--results_dir', help='results dir; auto-discover all repeat ckpts for the model')
    ap.add_argument('--model_name', default=None,
                    help='model tag for result filenames; default: config.ligand.models[0]')
    ap.add_argument('--split', default='base', help='only needed for pretrained_init struct branch; ignored for e2e')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch_eval', type=int, default=128)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--out_csv', default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_name = args.model_name or cfg['ligand']['models'][0]
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.ckpt:
        ckpts = [args.ckpt]
    else:
        ckpts = _discover_ckpts(args.results_dir, model_name)
        if not ckpts:
            print(f'[error] no {model_name}_*_*.pt found in {args.results_dir}', flush=True)
            return 1
    print(f'[eval] model={model_name}  device={device}  #ckpt={len(ckpts)}', flush=True)

    rows = {t: [] for t in args.test_datasets}
    for ci, ckpt_path in enumerate(ckpts):
        repeat = os.path.basename(ckpt_path)[:-3].rsplit('_', 1)[-1]
        repeat = int(repeat) if repeat.isdigit() else 0
        model = _build(cfg, model_name, device, args.split, repeat)
        nmiss, nunexp = _load_ckpt(model, ckpt_path, device)
        print(f'[eval] ({ci + 1}/{len(ckpts)}) loaded {os.path.basename(ckpt_path)}  '
              f'(missing={nmiss}, unexpected={nunexp})', flush=True)
        for t in args.test_datasets:
            m = _eval_one(model, device, args.data_root, t, args.batch_eval, args.num_workers)
            m['repeat'] = repeat
            rows[t].append(m)
            print(f'    {t:<14} N={m["n"]:>4}  RMSE={m["rmse"]:.4f}  MAE={m["mae"]:.4f}  '
                  f'SD={m["sd"]:.4f}  Pearson_r={m["r"]:.4f}', flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print('\n=== Summary (mean +/- std over checkpoints) ===', flush=True)
    csv_lines = ['test_set,repeat,rmse,mae,sd,r']
    for t, ms in rows.items():
        for m in ms:
            csv_lines.append(f'{t},{m["repeat"]},{m["rmse"]:.4f},{m["mae"]:.4f},'
                             f'{m["sd"]:.4f},{m["r"]:.4f}')
        if ms:
            arr = {k: np.array([m[k] for m in ms]) for k in ('rmse', 'mae', 'sd', 'r')}
            print(f'  {t:<14} '
                  f'RMSE={arr["rmse"].mean():.4f}+/-{arr["rmse"].std():.4f}  '
                  f'MAE={arr["mae"].mean():.4f}+/-{arr["mae"].std():.4f}  '
                  f'SD={arr["sd"].mean():.4f}+/-{arr["sd"].std():.4f}  '
                  f'r={arr["r"].mean():.4f}+/-{arr["r"].std():.4f}  (n_ckpt={len(ms)})', flush=True)
            csv_lines.append(f'{t},mean,{arr["rmse"].mean():.4f},{arr["mae"].mean():.4f},'
                             f'{arr["sd"].mean():.4f},{arr["r"].mean():.4f}')
            csv_lines.append(f'{t},std,{arr["rmse"].std():.4f},{arr["mae"].std():.4f},'
                             f'{arr["sd"].std():.4f},{arr["r"].std():.4f}')

    if args.out_csv:
        with open(args.out_csv, 'w') as f:
            f.write('\n'.join(csv_lines) + '\n')
        print(f'\n[eval] wrote -> {args.out_csv}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

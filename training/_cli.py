"""Shared CLI parsing and dispatch logic for the 5 train_*.py entry points, to avoid duplication."""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from training.train_runner import run_experiment, SplitSpec  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True,
                   help='path to yaml config (e.g. configs/frozen_t33.yaml)')
    p.add_argument('--data_root', type=str, default='data_processed_esm',
                   help='dataset root for PDBbindESMDataset')
    p.add_argument('--device', type=str, default=None,
                   help='override yaml device, e.g. cuda:0 / cuda:1')
    p.add_argument('--models', type=str, default=None,
                   help='comma-separated ligand models to run, '
                        'e.g. "GINConvNet,GATNet" (override yaml ligand.models)')
    p.add_argument('--no_resume', action='store_true',
                   help='disable auto-resume; will overwrite existing result CSVs')
    p.add_argument('--batch-train', type=int, default=None,
                   help='override yaml training.batch_size_train')
    p.add_argument('--batch-eval', type=int, default=None,
                   help='override yaml training.batch_size_eval')
    p.add_argument('--num-workers', type=int, default=None,
                   help='override yaml training.num_workers')
    return p


def parse_models(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    return [m.strip() for m in s.split(',') if m.strip()]


def run_from_cli(split: SplitSpec):
    args = build_arg_parser().parse_args()
    run_experiment(
        config_path=args.config,
        split=split,
        data_root=os.path.join(_ROOT, args.data_root),
        device_override=args.device,
        models_override=parse_models(args.models),
        resume=(not args.no_resume),
        batch_train_override=args.batch_train,
        batch_eval_override=args.batch_eval,
        num_workers_override=args.num_workers,
    )

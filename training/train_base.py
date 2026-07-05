"""Training entry for the base split: PDBbind v2020 minus CASF-2016, 9:1 train/valid;
external test on CASF-2016 + CSAR-HiQ.

Usage
----
python training/train_base.py --config configs/trifusion_prott5_u50.yaml
python training/train_base.py --config configs/trifusion_efficient_esm2.yaml --device cuda:1

Shared CLI flags are defined in training/_cli.py:
  --config       path to the yaml config
  --data_root    PyG dataset root (default data_processed_esm)
  --device       override yaml device
  --models       comma-separated ligand model subset
  --no_resume    disable auto-resume
"""
from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from training._cli import run_from_cli  # noqa: E402
from training.train_runner import SplitSpec  # noqa: E402


def main():
    split = SplitSpec(
        name='base',
        train_template='train_{split}_{repeat}',
        valid_template='valid_{split}_{repeat}',
        test_specs=[('CASF', 'CASF-2016'), ('CSAR', 'CSAR-HiQ')],
        n_repeats=5,
    )
    run_from_cli(split)


if __name__ == '__main__':
    main()

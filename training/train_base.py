"""Training entry for the base split: PDBbind v2020 minus CASF-2016, 9:1 train/valid;
external test on CASF-2016 (285) and CSAR-HiQ dedup (81; see data/splits/README.md)."""
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

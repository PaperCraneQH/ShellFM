"""Training entry for the holdout split (temporal extrapolation; single repeat)."""
from __future__ import annotations
import os, sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from training._cli import run_from_cli  # noqa: E402
from training.train_runner import SplitSpec  # noqa: E402

def main():
    split = SplitSpec(
        name='holdout',
        train_template='train_holdout',
        valid_template='valid_holdout',
        test_specs=[('test', 'test_holdout')],
        n_repeats=1,
    )
    run_from_cli(split)

if __name__ == '__main__':
    main()

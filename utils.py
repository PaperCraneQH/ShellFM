"""Shared utilities: Logger, set_seed, metrics, and the Evaluate class.

Adapted from the original GraphDTA utils.py + training_base.py (Logger / set_seed / cal_final_results);
without the create_data_PDBbind dependency (seq_cat / smiles_to_graph are not used here).
"""
import os
import random
from datetime import datetime, timedelta
from math import sqrt

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import LinearRegression


# ============================================================
#  Logger
# ============================================================
class Logger:
    """Simple logger that writes to both stdout and a file with timestamp prefixes.

    Matches the original GraphDTA training_base.py output style for fair comparison with legacy runs.
    """

    def __init__(self, log_file=None):
        self.log_file = log_file
        if log_file is not None:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            self._fh = open(log_file, 'a', encoding='utf-8')
        else:
            self._fh = None

    def _ts(self):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def log(self, msg='', with_ts=True, to_terminal=True):
        """Unified log entry.

        Parameters
        ----------
        to_terminal : bool
            When False, write to the log file only (no stdout). Used during training for
            high-frequency batch progress (kept in the log for debugging; terminal shows epoch summaries only).
        """
        line = f'[{self._ts()}] {msg}' if with_ts else msg
        if to_terminal:
            print(line, flush=True)
        if self._fh is not None:
            self._fh.write(line + '\n')
            self._fh.flush()

    def section(self, title, char='=', width=80, to_terminal=True):
        bar = char * width
        self.log('', with_ts=False, to_terminal=to_terminal)
        self.log(bar, with_ts=False, to_terminal=to_terminal)
        self.log(f'{title:^{width}}', with_ts=False, to_terminal=to_terminal)
        self.log(bar, with_ts=False, to_terminal=to_terminal)

    def subsection(self, title, char='-', width=80, to_terminal=True):
        bar = char * width
        self.log('', with_ts=False, to_terminal=to_terminal)
        self.log(bar, with_ts=False, to_terminal=to_terminal)
        self.log(title, with_ts=False, to_terminal=to_terminal)
        self.log(bar, with_ts=False, to_terminal=to_terminal)

    def close(self):
        if self._fh is not None:
            self._fh.close()


def fmt_secs(s):
    return str(timedelta(seconds=int(s)))


# ============================================================
#  Reproducibility
# ============================================================
def set_seed(seed=42, deterministic=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
#  Metrics
# ============================================================
def rmse(y, f):
    return sqrt(((y - f) ** 2).mean(axis=0))


def mse(y, f):
    return ((y - f) ** 2).mean(axis=0)


def pearson(y, f):
    return np.corrcoef(y, f)[0, 1]


def spearman(y, f):
    return stats.spearmanr(y, f)[0]


class Evaluate:
    """Equivalent to the original GraphDTA Evaluate: RMSE / MAE / SD / Pearson r."""

    def _rmse(self, y, p):
        return sqrt(((y - p) ** 2).mean(axis=0))

    def _mae(self, y, p):
        return (np.abs(y - p)).mean()

    def _sd(self, y, p):
        p, y = p.reshape(-1, 1), y.reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(p, y)
        y_ = lr.predict(p)
        return (((y - y_) ** 2).sum() / (len(y) - 1)) ** 0.5

    def _pearson(self, y, p):
        return np.corrcoef(y, p)[0, 1]

    def evaluate(self, y_true, pred):
        return self._rmse(y_true, pred), self._mae(y_true, pred), \
               self._pearson(y_true, pred), self._sd(y_true, pred)


# ============================================================
#  CSV aggregation: 5 repeats -> avg / std
# ============================================================
def cal_final_results(result_file, n_repeat, logger=None):
    res = pd.read_csv(result_file)
    metrics = ['rmse', 'mae', 'sd', 'r']
    res_ls = {m: [] for m in metrics}
    try:
        for m in metrics:
            res_ls[m] = [float(res[res['repeat'] == i][m].values[0]) for i in range(n_repeat)]
    except Exception:
        for m in metrics:
            res_ls[m] = [float(res[res['repeat'] == str(i)][m].values[0]) for i in range(n_repeat)]
    avg_res = [round(float(np.mean(res_ls[m])), 4) for m in metrics]
    std_res = [round(float(np.std(res_ls[m])), 4) for m in metrics]
    res.loc[len(res)] = ['avg'] + avg_res
    res.loc[len(res)] = ['std'] + std_res
    res.to_csv(result_file, index=False)

    if logger is not None:
        logger.subsection(f'Aggregated over {n_repeat} repeats  ->  {os.path.basename(result_file)}')
        logger.log(f'  {"metric":<8}{"mean":>12}{"std":>12}', with_ts=False)
        for m, a, s in zip(metrics, avg_res, std_res):
            logger.log(f'  {m:<8}{a:>12.4f}{s:>12.4f}', with_ts=False)

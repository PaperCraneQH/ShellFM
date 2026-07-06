"""General training runner invoked by the five split entry scripts."""
from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader  # plain torch DataLoader for custom collate_fn

# Project root
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data import (  # noqa: E402
    PDBbindESMDataset, esm_collate_fn, LengthBucketSampler,
)
from models import build_model, setup_struct_branch  # noqa: E402
from utils import Evaluate, Logger, cal_final_results, fmt_secs, mse, set_seed  # noqa: E402


# ==============================================================
#  Split specification
# ==============================================================
@dataclass
class SplitSpec:
    """Describes the training workflow for one data split."""
    name: str                                   # 'base' | 'random' | ...
    train_template: str                         # 'train_{split}_{repeat}'
    valid_template: str                         # 'valid_{split}_{repeat}'
    # Test-set list: each entry is (tag, dataset_name_template_or_fixed)
    test_specs: List[tuple]                     # [('CASF', 'CASF-2016'), ...]
    n_repeats: int = 5

    def train_name(self, repeat: int) -> str:
        return self.train_template.format(split=self.name, repeat=repeat)

    def valid_name(self, repeat: int) -> str:
        return self.valid_template.format(split=self.name, repeat=repeat)

    def test_name(self, template: str, repeat: int) -> str:
        # holdout uses `test_{split}` (no {repeat}); old logic only formatted when {repeat}
        # was present, treating literal `test_{split}` as the dataset name and failing on missing .pt.
        if '{split}' in template or '{repeat}' in template:
            return template.format(split=self.name, repeat=repeat)
        return template


# SplitSpec registry for all five splits (also used by [tools/evaluate.py](../tools/evaluate.py))
SPLIT_SPECS: Dict[str, 'SplitSpec'] = {
    'base': SplitSpec(
        name='base',
        train_template='train_{split}_{repeat}',
        valid_template='valid_{split}_{repeat}',
        # base has no in-fold test; external CASF (285) + CSAR-HiQ dedup (81, see data/splits/README.md)
        test_specs=[('CASF', 'CASF-2016'), ('CSAR', 'CSAR-HiQ')],
        n_repeats=5,
    ),
    'random': SplitSpec(
        name='random',
        train_template='train_{split}_{repeat}',
        valid_template='valid_{split}_{repeat}',
        test_specs=[('test', 'test_{split}_{repeat}')],
        n_repeats=5,
    ),
    'holdout': SplitSpec(
        name='holdout',
        train_template='train_{split}',
        valid_template='valid_{split}',
        test_specs=[('test', 'test_{split}')],
        n_repeats=5,  # same data, 5 different seeds
    ),
    'scaffold': SplitSpec(
        name='scaffold',
        train_template='train_{split}_{repeat}',
        valid_template='valid_{split}_{repeat}',
        test_specs=[('test', 'test_{split}_{repeat}')],
        n_repeats=5,
    ),
    'seq_identity': SplitSpec(
        name='seq_identity',
        train_template='train_{split}_{repeat}',
        valid_template='valid_{split}_{repeat}',
        test_specs=[('test', 'test_{split}_{repeat}')],
        n_repeats=5,
    ),
    # GEMS CleanSplit built-in 5-fold CV. Only meaningful when data_family=cleansplit
    # Usage: python training/train_gems_5fold.py --config <yaml> --data_root data_processed_esm_cleansplit
    # Files: cold_start_cleansplit/gems_5fold/{train,valid}_gems_5fold_{0..4}.csv
    # External test remains CASF-2016 (CleanSplit ensures no structural leakage train ↔ CASF-2016)
    'gems_5fold': SplitSpec(
        name='gems_5fold',
        train_template='train_{split}_{repeat}',
        valid_template='valid_{split}_{repeat}',
        test_specs=[('CASF', 'CASF-2016')],
        n_repeats=5,
    ),
    # EHIGN paper main experiment: fixed train/valid + test2013/2016/2019, 3 repeats (same as train.py)
    'paper': SplitSpec(
        name='paper',
        train_template='train_paper',
        valid_template='valid_paper',
        test_specs=[
            ('test2013', 'test2013'),
            ('test2016', 'test2016'),
            ('test2019', 'test2019'),
        ],
        n_repeats=3,
    ),
}


def get_split_spec(name: str) -> 'SplitSpec':
    if name not in SPLIT_SPECS:
        raise KeyError(f'unknown split {name!r}. Available: {list(SPLIT_SPECS.keys())}')
    return SPLIT_SPECS[name]


# ==============================================================
#  Train / Eval epoch
# ==============================================================
def _resolve_amp_dtype(name) -> Optional[torch.dtype]:
    """yaml `training.amp_dtype` -> torch.dtype | None.

    Supported: 'bf16' / 'bfloat16' / 'fp16' / 'float16' / 'none' / None / ''
    Default None (no autocast; preserves legacy behavior)
    """
    if name is None:
        return None
    s = str(name).strip().lower()
    if s in ('', 'none', 'null', 'off', 'disable', 'false'):
        return None
    if s in ('bf16', 'bfloat16'):
        return torch.bfloat16
    if s in ('fp16', 'float16', 'half'):
        return torch.float16
    raise ValueError(f'unknown amp_dtype: {name!r}. '
                     f'Supported: bf16/bfloat16/fp16/float16/none')


def _unpack_fusion_output(output):
    """Fusion head may return (main, [aux1, aux2, ...]) for deep supervision, or a single tensor.

    Returns (main_pred, aux_list). aux_list is None when there is no deep supervision.
    """
    if isinstance(output, (tuple, list)):
        return output[0], (output[1] if len(output) > 1 else None)
    return output, None


def train_one_epoch(model, device, loader, optimizer, loss_fn,
                    epoch: int, log_interval: int, logger: Logger,
                    grad_clip: Optional[float] = 1.0,
                    amp_dtype: Optional[torch.dtype] = None,
                    grad_scaler: Optional[torch.cuda.amp.GradScaler] = None,
                    trainable_params: Optional[List[torch.nn.Parameter]] = None,
                    aux_loss_weight: float = 0.0):
    """Run one training epoch.

    Optimizations (significant in LoRA mode):
      1. When `amp_dtype` is set, wrap entire forward + loss in autocast; backward pairs
         automatically. bf16 needs no GradScaler; fp16 needs one (pass grad_scaler).
      2. Train metrics (RMSE/r) accumulate on GPU tensors; single CPU transfer at log_interval,
         avoiding per-batch GPU->CPU sync stalls.
      3. `trainable_params` pre-cached by caller (avoids per-batch list comprehension).

    Returns
    -------
    (avg_loss, train_rmse, train_pearson_r, samples_per_sec)
        - train_rmse / train_pearson_r from epoch-accumulated preds and labels
        - samples_per_sec for ETA estimation
    """
    model.train()
    # frozen / adapter-only: backbone eval disables dropout; adapter branches set train() in forward
    prot_bb = getattr(model.prot, 'esm', None) or getattr(model.prot, 'backbone', None)
    if prot_bb is not None and (
            not getattr(model.prot, 'has_trainable_esm', False)
            or getattr(model.prot, 'adapter_only', False)):
        prot_bb.eval()
    lig_lm = getattr(model.lig, 'lm', None)
    if lig_lm is not None and getattr(model.lig, 'adapter_only', False):
        lig_lm.eval()

    n_samples = len(loader.dataset)
    n_batches = len(loader)
    running_sum = 0.0
    running_n = 0
    # Accumulate preds and labels on GPU (avoid per-batch sync; single transfer at epoch end)
    pred_chunks: List[torch.Tensor] = []
    label_chunks: List[torch.Tensor] = []

    use_amp = amp_dtype is not None and device.type == 'cuda'
    use_scaler = use_amp and amp_dtype == torch.float16 and grad_scaler is not None

    # Pre-cache trainable params (avoid list comprehension every batch)
    if trainable_params is None:
        trainable_params = [p for p in model.parameters() if p.requires_grad]

    t_start = time.time()
    for batch_idx, (pyg_batch, seqs, smiles, pdb_codes) in enumerate(loader):
        pyg_batch = pyg_batch.to(device, non_blocking=True)
        targets = pyg_batch.y.view(-1, 1).float().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                output = model(pyg_batch, seqs, smiles, pdb_codes)
                output, aux_outs = _unpack_fusion_output(output)
                loss = loss_fn(output, targets)
                if aux_outs and aux_loss_weight > 0.0:
                    for a in aux_outs:
                        loss = loss + aux_loss_weight * loss_fn(a, targets)
        else:
            output = model(pyg_batch, seqs, smiles, pdb_codes)
            output, aux_outs = _unpack_fusion_output(output)
            loss = loss_fn(output, targets)
            if aux_outs and aux_loss_weight > 0.0:
                for a in aux_outs:
                    loss = loss + aux_loss_weight * loss_fn(a, targets)

        if use_scaler:
            grad_scaler.scale(loss).backward()
            if grad_clip is not None:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(grad_clip))
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(grad_clip))
            optimizer.step()

        bsz = targets.size(0)
        running_sum += loss.item() * bsz   # .item() needed for Python scalar accumulation
        running_n += bsz
        # Accumulate on GPU (detach + float to fp32 for metric precision)
        pred_chunks.append(output.detach().float().view(-1))
        label_chunks.append(targets.detach().float().view(-1))

        if batch_idx % log_interval == 0 or batch_idx == n_batches - 1:
            # Use running_n for batch_sampler (when loader.batch_size=None,
            # old (batch_idx+1)*loader.batch_size raises TypeError)
            seen = running_n
            pct = 100. * (batch_idx + 1) / n_batches
            elapsed = time.time() - t_start
            sps = running_n / max(elapsed, 1e-6)
            remaining_batches = n_batches - (batch_idx + 1)
            eta_epoch = elapsed * remaining_batches / max(batch_idx + 1, 1)
            # Single concat + transfer (vs per-batch cpu().numpy())
            preds_so_far = torch.cat(pred_chunks).cpu().numpy()
            labels_so_far = torch.cat(label_chunks).cpu().numpy()
            tr_rmse = float(np.sqrt(((preds_so_far - labels_so_far) ** 2).mean()))
            if preds_so_far.std() > 1e-8 and labels_so_far.std() > 1e-8:
                tr_r = float(np.corrcoef(preds_so_far, labels_so_far)[0, 1])
            else:
                tr_r = float('nan')

            logger.log(
                f'  [epoch {epoch:>3}] batch {batch_idx + 1:>4}/{n_batches} '
                f'({seen:>6}/{n_samples} | {pct:5.1f}%)  '
                f'batch_loss={loss.item():.4f}  avg_loss={running_sum / max(running_n, 1):.4f}  '
                f'tr_RMSE={tr_rmse:.4f}  tr_r={tr_r:.4f}  '
                f'speed={sps:.0f} sps  eta(epoch)={fmt_secs(eta_epoch)}',
                to_terminal=False,
            )

    avg_loss = running_sum / max(running_n, 1)
    if pred_chunks:
        preds = torch.cat(pred_chunks).cpu().numpy()
        labels = torch.cat(label_chunks).cpu().numpy()
        tr_rmse = float(np.sqrt(((preds - labels) ** 2).mean()))
        if preds.std() > 1e-8 and labels.std() > 1e-8:
            tr_r = float(np.corrcoef(preds, labels)[0, 1])
        else:
            tr_r = float('nan')
    else:
        tr_rmse = float('nan')
        tr_r = float('nan')
    sps = running_n / max(time.time() - t_start, 1e-6)
    return avg_loss, tr_rmse, tr_r, sps


@torch.no_grad()
def predict(model, device, loader, amp_dtype: Optional[torch.dtype] = None):
    """Eval pass.

    When amp_dtype is set, wrap entire forward in autocast for faster LoRA eval.
    GPU tensor accumulation; single cat + .cpu() at end (vs per-batch sync).
    """
    model.eval()
    use_amp = amp_dtype is not None and device.type == 'cuda'
    pred_chunks: List[torch.Tensor] = []
    label_chunks: List[torch.Tensor] = []
    for pyg_batch, seqs, smiles, pdb_codes in loader:
        pyg_batch = pyg_batch.to(device, non_blocking=True)
        if use_amp:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                out = model(pyg_batch, seqs, smiles, pdb_codes)
        else:
            out = model(pyg_batch, seqs, smiles, pdb_codes)
        out, _ = _unpack_fusion_output(out)
        pred_chunks.append(out.detach().float().view(-1))
        label_chunks.append(pyg_batch.y.view(-1).detach().float())
    if not pred_chunks:
        return np.array([]), np.array([])
    preds = torch.cat(pred_chunks).cpu().numpy()
    labels = torch.cat(label_chunks).cpu().numpy()
    return labels, preds


def evaluate_and_log(metric: Evaluate, G, P, name: str, logger: Logger):
    rmse, mae, r, sd = metric.evaluate(G, P)
    logger.log(f'  >>> {name:<18} '
               f'RMSE={rmse:.4f}  MAE={mae:.4f}  SD={sd:.4f}  Pearson_r={r:.4f}  '
               f'(N={len(G)})')
    return rmse, mae, sd, r


# ==============================================================
#  Optimizer with parameter groups
# ==============================================================
def make_optimizer(model, base_lr: float, esm_lr: float, weight_decay: float,
                   lig_lr: Optional[float] = None,
                   struct_lr: Optional[float] = None):
    """Parameter groups:
      - prot.esm.*                         -> esm_lr
      - lig.lm.*                           -> lig_lr (default=base_lr)
      - plitext.backbone.* (structure GNN) -> struct_lr (default=base_lr)
      - rest (proj / fusion / ...)         -> base_lr
    """
    esm_params, lig_params, struct_params, other_params = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith('prot.esm'):
            esm_params.append(p)
        elif n.startswith('lig.lm'):
            lig_params.append(p)
        elif n.startswith('plitext.backbone'):
            struct_params.append(p)
        else:
            other_params.append(p)

    groups = [{'params': other_params, 'lr': base_lr, 'weight_decay': weight_decay}]
    if esm_params:
        groups.append({'params': esm_params, 'lr': esm_lr, 'weight_decay': weight_decay})
    if lig_params:
        groups.append({'params': lig_params,
                       'lr': (lig_lr if lig_lr is not None else base_lr),
                       'weight_decay': weight_decay})
    if struct_params:
        groups.append({'params': struct_params,
                       'lr': (struct_lr if struct_lr is not None else base_lr),
                       'weight_decay': weight_decay})
    return torch.optim.AdamW(groups)


# ==============================================================
#  LR scheduler factory
# ==============================================================
def make_scheduler(optimizer, sched_cfg: Optional[Dict[str, Any]]):
    """Build LR scheduler from yaml ``training.scheduler`` section.

    Returns
    -------
    (scheduler, step_kind)
      - scheduler : torch.optim.lr_scheduler instance, or None
      - step_kind : 'plateau' / 'epoch' / None
            'plateau' = call scheduler.step(val_metric) at end of each epoch
            'epoch'   = call scheduler.step() (no args) at end of each epoch
            None      = no scheduler
    Default (empty sched_cfg / name=none): fixed lr (legacy behavior).
    """
    if not sched_cfg:
        return None, None
    name = str(sched_cfg.get('name', '')).strip().lower()
    if name in ('', 'none', 'null', 'off', 'disable'):
        return None, None

    if name in ('reduce_on_plateau', 'plateau', 'reducelronplateau'):
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        sched = ReduceLROnPlateau(
            optimizer,
            mode=str(sched_cfg.get('mode', 'min')),
            factor=float(sched_cfg.get('factor', 0.5)),
            patience=int(sched_cfg.get('patience', 5)),
            min_lr=float(sched_cfg.get('min_lr', 0.0)),
            cooldown=int(sched_cfg.get('cooldown', 0)),
            threshold=float(sched_cfg.get('threshold', 1e-4)),
            threshold_mode=str(sched_cfg.get('threshold_mode', 'rel')),
        )
        return sched, 'plateau'

    if name in ('cosine', 'cosineannealing', 'cosineannealinglr'):
        from torch.optim.lr_scheduler import CosineAnnealingLR
        sched = CosineAnnealingLR(
            optimizer,
            T_max=int(sched_cfg.get('t_max', 200)),
            eta_min=float(sched_cfg.get('min_lr', 0.0)),
        )
        return sched, 'epoch'

    raise ValueError(f'unknown scheduler name: {name!r}. '
                     f'Supported: reduce_on_plateau / cosine')


# ==============================================================
#  JSON serialization helpers
# ==============================================================
def _to_jsonable(obj):
    """Recursively convert numpy scalars/arrays/containers to native Python types
    for ``json.dump`` (numpy.float32/64 are not JSON-serializable by default)."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj


# ==============================================================
#  Checkpoint helpers (for true mid-training resume)
# ==============================================================
@dataclass
class TrainingState:
    """Scalar state to persist during one (model, repeat) training run."""
    start_epoch: int = 1
    best_val_mse: float = float('inf')
    best_epoch: int = -1
    best_val_metrics: Optional[tuple] = None       # (rmse, mae, sd, r)
    no_improve: int = 0
    epoch_times: List[float] = field(default_factory=list)
    best_state_dict: Optional[Dict[str, Any]] = None   # weight snapshot at best epoch
    # Training curve: one dict per epoch for plotting
    history: List[Dict[str, Any]] = field(default_factory=list)


# Pretrained backbone prefixes: frozen weights reloaded from HF, excluded from checkpoint.
#   prot.esm.*  -> ESM-2 protein backbone
#   lig.lm.*    -> ChemBERTa ligand backbone
_PRETRAINED_BACKBONE_PREFIXES = ('prot.esm.', 'lig.lm.')


def _is_backbone_key(k: str) -> bool:
    return any(k.startswith(pfx) for pfx in _PRETRAINED_BACKBONE_PREFIXES)


def _extract_savable_state(model) -> Dict[str, Any]:
    """Strip frozen pretrained backbones (ESM-2 / ChemBERTa) from model.state_dict().

    Keeps trainable params (incl. LoRA adapters) and all non-backbone parts (GNN / fusion / proj / buffers).
    """
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    out: Dict[str, Any] = {}
    for k, v in model.state_dict().items():
        is_frozen_backbone = _is_backbone_key(k) and k not in trainable_names
        if is_frozen_backbone:
            continue
        out[k] = v
    return out


def _capture_rng_state() -> Dict[str, Any]:
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(rng: Optional[Dict[str, Any]]) -> None:
    if not rng:
        return
    try:
        if 'python' in rng:
            random.setstate(rng['python'])
        if 'numpy' in rng:
            np.random.set_state(rng['numpy'])
        if 'torch' in rng:
            torch.set_rng_state(rng['torch'])
        if rng.get('cuda') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng['cuda'])
    except Exception:
        # RNG restore failure does not affect training correctness, only reproducibility
        pass


def _save_checkpoint(path: str, *, model, optimizer, scheduler, epoch: int,
                     state: TrainingState, done_training: bool) -> None:
    """Write full checkpoint to disk.

    Payload layout:
      state_dict          <- best weights (read at test; compatible with legacy files)
      latest_state_dict   <- weights at end of current epoch (for resume)
      optimizer           <- AdamW state
      scheduler           <- LR scheduler state (None if no scheduler)
      epoch               <- completed epoch index
      best_val_mse / best_epoch / best_val_metrics / no_improve / epoch_times
      rng_state           <- python/numpy/torch/cuda RNG snapshot
      done_training       <- True if early-stop or max-epoch reached; next run goes to test
    """
    latest = _extract_savable_state(model)
    payload = {
        'state_dict': state.best_state_dict if state.best_state_dict is not None else latest,
        'latest_state_dict': latest,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'epoch': int(epoch),
        'best_val_mse': float(state.best_val_mse),
        'best_epoch': int(state.best_epoch),
        'best_val_metrics': state.best_val_metrics,
        'no_improve': int(state.no_improve),
        'epoch_times': list(state.epoch_times),
        'history': list(state.history),
        'rng_state': _capture_rng_state(),
        'done_training': bool(done_training),
        'format_version': 4,
    }
    # Atomic write: .tmp then rename to avoid truncated ckpt on crash
    tmp = path + '.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _try_load_checkpoint(path: str, device, logger: Logger,
                         model, optimizer, scheduler=None) -> Optional[TrainingState]:
    """Try loading full checkpoint. On success returns TrainingState and injects
    weights/optimizer/scheduler; legacy format or missing fields -> None (train from scratch)."""
    if not os.path.isfile(path):
        return None
    try:
        ckpt = torch.load(path, map_location=device)
    except Exception as e:
        logger.log(f'[resume][warn] failed to load {os.path.basename(path)}: {e}; '
                   f'will retrain from scratch')
        return None
    if 'epoch' not in ckpt or 'latest_state_dict' not in ckpt:
        # Legacy format: best state_dict only, no resume info
        logger.log(f'[resume][warn] {os.path.basename(path)} is legacy format '
                   f'(no resume info); will retrain from scratch (existing best weights ignored)')
        return None

    state = TrainingState()
    state.best_val_mse = float(ckpt.get('best_val_mse', float('inf')))
    state.best_epoch = int(ckpt.get('best_epoch', -1))
    state.best_val_metrics = ckpt.get('best_val_metrics')
    state.no_improve = int(ckpt.get('no_improve', 0))
    state.epoch_times = list(ckpt.get('epoch_times', []))
    state.history = list(ckpt.get('history', []))
    state.best_state_dict = ckpt.get('state_dict')
    state.start_epoch = int(ckpt['epoch']) + 1
    state._done_training = bool(ckpt.get('done_training', False))  # type: ignore[attr-defined]

    # ---- Inject weights / optimizer / RNG into runtime ----
    try:
        missing, unexpected = model.load_state_dict(ckpt['latest_state_dict'], strict=False)
        non_backbone_missing = [k for k in missing if not _is_backbone_key(k)]
        if non_backbone_missing:
            logger.log(f'[resume][warn] non-backbone missing keys when loading latest: '
                       f'{non_backbone_missing[:5]}...')
        if unexpected:
            logger.log(f'[resume][warn] unexpected keys when loading latest: {len(unexpected)}')
    except Exception as e:
        logger.log(f'[resume][warn] model state load failed: {e}; will retrain from scratch')
        return None

    if 'optimizer' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
        except Exception as e:
            logger.log(f'[resume][warn] optimizer state load failed: {e}; '
                       f'will keep freshly-built optimizer')

    sched_state = ckpt.get('scheduler')
    if scheduler is not None and sched_state is not None:
        try:
            scheduler.load_state_dict(sched_state)
        except Exception as e:
            logger.log(f'[resume][warn] scheduler state load failed: {e}; '
                       f'will keep freshly-built scheduler')
    elif scheduler is not None and sched_state is None:
        logger.log('[resume][warn] checkpoint has no scheduler state '
                   '(pre-format_version=4); scheduler will start from fresh state.')

    _restore_rng_state(ckpt.get('rng_state'))
    return state


# ==============================================================
#  Result CSV helpers (for resume / skip-existing)
# ==============================================================
def _completed_repeats_in_csv(csv_path: str) -> set:
    """Read result CSV and return set of completed repeat ids (int).

    CSV may have trailing 'avg' / 'std' rows (appended after all repeats);
    that means the experiment group is **fully done** -> return special set {-1}.
    """
    if not os.path.isfile(csv_path):
        return set()
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        if 'repeat' not in df.columns:
            return set()
        vals = df['repeat'].astype(str).tolist()
        if 'avg' in vals or 'std' in vals:
            return {-1}  # aggregated -> entire group complete
        done = set()
        for v in vals:
            try:
                done.add(int(v))
            except ValueError:
                pass
        return done
    except Exception:
        return set()


# ==============================================================
#  Main entry per (config, split)
# ==============================================================
def run_experiment(config_path: str, split: SplitSpec,
                   data_root: str = 'data_processed_esm',
                   device_override: Optional[str] = None,
                   models_override: Optional[List[str]] = None,
                   resume: bool = True,
                   batch_train_override: Optional[int] = None,
                   batch_eval_override: Optional[int] = None,
                   num_workers_override: Optional[int] = None):
    """Run one (config, split) experiment.

    Parameters
    ----------
    config_path : str
        Path to yaml config.
    split : SplitSpec
        Train / valid / test dataset templates.
    data_root : str
        PDBbindESMDataset root; default data_processed_esm.
    device_override : Optional[str]
        CLI override for yaml device, e.g. 'cuda:1'.
    models_override : Optional[List[str]]
        CLI override for yaml ligand.models, e.g. ['GINConvNet','GATNet'].
        For dual-GPU: split 4 GNNs into two groups on two cards.
    resume : bool
        When True (default): skip repeats already in result CSV;
        skip entire (model, split) if group already aggregated (avg/std rows).
    """
    # ---- Load config ----
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    exp_cfg = cfg['experiment']
    train_cfg = cfg['training']

    esm_mode = exp_cfg.get('esm_mode', 'frozen')
    lr = float(train_cfg.get('lr', 1e-4))
    lr_esm = float(train_cfg.get('lr_esm', lr))
    lr_lig = float(train_cfg.get('lr_lig', lr))   # ChemBERTa LoRA lr (default=base)
    lr_struct = float(train_cfg.get('lr_struct', lr))
    wd = float(train_cfg.get('weight_decay', 1e-4))
    batch_train = int(batch_train_override or train_cfg.get('batch_size_train', 32))
    batch_eval = int(batch_eval_override or train_cfg.get('batch_size_eval', batch_train))
    num_epochs = int(train_cfg.get('epochs', 200))
    patience = int(train_cfg.get('early_stop_patience', 15))
    sched_cfg = train_cfg.get('scheduler')
    n_repeats = int(train_cfg.get('n_repeats', split.n_repeats))
    log_interval = int(train_cfg.get('log_interval', 20))
    num_workers = int(num_workers_override if num_workers_override is not None
                       else train_cfg.get('num_workers', 4))
    pin_memory = bool(train_cfg.get('pin_memory', True))
    persistent_workers = bool(train_cfg.get('persistent_workers', True)) and num_workers > 0
    prefetch_factor = train_cfg.get('prefetch_factor')   # None = DataLoader default (2)
    if prefetch_factor is not None:
        prefetch_factor = int(prefetch_factor)
    # Mixed precision (bf16 recommended for LoRA; frozen default off for stability)
    amp_dtype = _resolve_amp_dtype(train_cfg.get('amp_dtype'))
    # Length-bucket batches by protein length (LoRA recommended; ~1.3-1.5x less padding waste)
    bucket_by_length = bool(train_cfg.get('bucket_by_length', False))
    bucket_count = train_cfg.get('bucket_count')      # None = sampler auto-select
    if bucket_count is not None:
        bucket_count = int(bucket_count)
    device_str = device_override or train_cfg.get('device', 'cuda:0')
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available() and device.type == 'cuda':
        frac = os.environ.get('CUDA_MEMORY_FRACTION')
        if frac:
            idx = device.index if device.index is not None else torch.cuda.current_device()
            torch.cuda.set_per_process_memory_fraction(float(frac), idx)

    ligand_models = list(models_override) if models_override else list(cfg['ligand']['models'])
    lig_type = str(cfg.get('ligand', {}).get('type', 'chemberta')).lower()

    # Output directory uses experiment.tag from yaml.
    exp_tag = exp_cfg.get('tag') or esm_mode

    # ---- Output dir ----
    save_root = os.path.join(_ROOT, 'results',
                             f'{split.name}_{exp_tag}_lr{lr:g}')
    os.makedirs(save_root, exist_ok=True)

    # Log filename distinguishes model subset for this run:
    #   single GPU with all 4 yaml models -> train.log (backward compatible)
    #   dual-GPU dispatcher / --models subset -> train_<modelA>_<modelB>.log
    # Parallel workers no longer write to the same file.
    if models_override:
        log_filename = 'train_' + '_'.join(models_override) + '.log'
    else:
        log_filename = 'train.log'
    log_path = os.path.join(save_root, log_filename)
    logger = Logger(log_path)

    logger.section(
        f'GraphDTA-ESM  |  split={split.name}  |  tag={exp_tag}  |  '
        f'esm_mode={esm_mode}  |  lr={lr:g}',
        char='#',
    )
    logger.log(f'config              : {config_path}')
    logger.log(f'device              : {device}')
    logger.log(f'train batch size    : {batch_train}')
    logger.log(f'eval  batch size    : {batch_eval}')
    logger.log(f'learning rate       : {lr:g}  (esm_lr={lr_esm:g}, lig_lr={lr_lig:g})')
    logger.log(f'ligand encoder type : {lig_type}')
    logger.log(f'weight_decay        : {wd:g}')
    logger.log(f'max epochs          : {num_epochs}')
    logger.log(f'early-stop patience : {patience}')
    if sched_cfg:
        sched_name = str(sched_cfg.get('name', 'none'))
        if sched_name.lower() in ('', 'none', 'null', 'off', 'disable'):
            logger.log(f'lr scheduler        : (disabled)')
        else:
            extra = {k: v for k, v in sched_cfg.items() if k != 'name'}
            logger.log(f'lr scheduler        : {sched_name}  {extra}')
    else:
        logger.log(f'lr scheduler        : (none, constant lr)')
    logger.log(f'repeats per model   : {n_repeats}')
    logger.log(f'num_workers         : {num_workers}  '
               f'(prefetch_factor={prefetch_factor if prefetch_factor is not None else "default(2)"})')
    logger.log(f'amp_dtype           : {amp_dtype if amp_dtype is not None else "off (fp32)"}')
    logger.log(f'bucket_by_length    : {bucket_by_length}'
               f'{f" (num_buckets={bucket_count})" if bucket_by_length and bucket_count else ""}')
    logger.log(f'log file            : {log_path}')
    logger.log(f'results dir         : {save_root}')
    logger.log(f'ligand models       : {ligand_models}')

    # ---- cuDNN benchmark ----
    torch.backends.cudnn.benchmark = True

    overall_start = time.time()

    for m_idx, model_name in enumerate(ligand_models):
        logger.section(f'MODEL {m_idx + 1}/{len(ligand_models)} : {model_name}', char='#')

        # One CSV per (model, test_tag); append when resume=True instead of overwrite
        result_files = {}
        already_done = {}
        for tag, _tpl in split.test_specs:
            rf = os.path.join(save_root, f'result_{model_name}_{split.name}_{tag}.csv')
            done = _completed_repeats_in_csv(rf) if resume else set()
            already_done[tag] = done
            if not resume or not os.path.isfile(rf):
                with open(rf, 'w') as f:
                    f.write('repeat,rmse,mae,sd,r\n')
            result_files[tag] = rf

        # Group already aggregated (avg/std) -> skip entire model
        all_aggregated = all((-1 in already_done[tag]) for tag in already_done)
        if resume and all_aggregated and already_done:
            logger.log(f'[resume] {model_name}: all test sets aggregated (avg/std present), skipping.')
            continue

        for repeat in range(n_repeats):
            # ---- Resume skip: only when every test tag has this repeat recorded ----
            if resume:
                already_repeat = all(
                    (repeat in already_done[tag]) or (-1 in already_done[tag])
                    for tag in already_done
                )
                if already_repeat:
                    logger.log(f'[resume] {model_name} repeat {repeat} already done '
                               f'(all test tags), skipping.')
                    continue

            seed = 42 + 2 ** repeat
            set_seed(seed)
            logger.section(
                f'{model_name}  |  repeat {repeat + 1}/{n_repeats}  |  seed={seed}',
                char='='
            )

            # ---- Data ----
            t0 = time.time()
            train_ds = PDBbindESMDataset(root=data_root, dataset=split.train_name(repeat))
            valid_ds = PDBbindESMDataset(root=data_root, dataset=split.valid_name(repeat))
            test_loaders = {}

            loader_kwargs = dict(
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                collate_fn=esm_collate_fn,
            )
            if prefetch_factor is not None and num_workers > 0:
                loader_kwargs['prefetch_factor'] = prefetch_factor

            # train_loader: length-bucketing only on train (eval/test use sequential batch;
            # no param updates there, padding waste is marginal vs total cost).
            if bucket_by_length:
                train_sampler = LengthBucketSampler(
                    train_ds, batch_size=batch_train,
                    num_buckets=bucket_count, shuffle=True,
                    drop_last=False, seed=seed,
                )
                train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                                          **loader_kwargs)
            else:
                train_loader = DataLoader(train_ds, batch_size=batch_train,
                                          shuffle=True, drop_last=False,
                                          **loader_kwargs)
            valid_loader = DataLoader(valid_ds, batch_size=batch_eval,
                                      shuffle=False, **loader_kwargs)

            test_sizes = {}
            for tag, tpl in split.test_specs:
                ds = PDBbindESMDataset(root=data_root, dataset=split.test_name(tpl, repeat))
                test_loaders[tag] = DataLoader(ds, batch_size=batch_eval,
                                               shuffle=False, **loader_kwargs)
                test_sizes[tag] = len(ds)

            size_str = '  '.join(f'#test({k})={v:>4}' for k, v in test_sizes.items())
            logger.log(f'#train={len(train_ds):>6}  '
                       f'#valid={len(valid_ds):>6}  '
                       f'{size_str}  '
                       f'(loaded in {time.time() - t0:.1f}s)')

            # ---- Model ----
            model = build_model(cfg, ligand_model_name=model_name).to(device)
            pli_cfg = cfg.get('plitext') or {}
            setup_struct_branch(model, pli_cfg, split=split.name, repeat=repeat)
            n_total = sum(p.numel() for p in model.parameters())
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.log(f'model={model_name}  '
                       f'total_params={n_total:,}  trainable_params={n_train:,}')

            optimizer = make_optimizer(model, base_lr=lr, esm_lr=lr_esm,
                                       weight_decay=wd, lig_lr=lr_lig,
                                       struct_lr=lr_struct)
            scheduler, sched_step_kind = make_scheduler(optimizer, sched_cfg)
            # 2026-05-21: configurable loss factory, MSE + ranking auxiliary
            # (notes/RANKING_LOSS_INTEGRATION_2026-05-21.md). Missing training.loss
            # falls back to pure MSE, fully backward compatible with old yaml.
            from training.losses import make_loss_fn  # noqa: E402
            loss_fn = make_loss_fn(cfg['training'].get('loss'))
            # Deep supervision: gated/MoE fusion returns (main, [aux...]);
            # apply aux_loss_weight * loss_fn(aux, target) to each aux head.
            aux_loss_weight = float((cfg.get('fusion') or {}).get('aux_weight', 0.0))
            metric = Evaluate()
            # GradScaler only for fp16; bf16 range (E8M7) avoids underflow
            grad_scaler = (
                torch.cuda.amp.GradScaler() if amp_dtype == torch.float16 else None
            )
            # Pre-cache trainable params (avoid per-batch model.parameters() scan in train_one_epoch)
            trainable_params = [p for p in model.parameters() if p.requires_grad]

            model_file = os.path.join(
                save_root, f'{model_name}_{split.name}_{exp_tag}_{repeat}.pt')

            # ---- Training state: default fresh start; resume from ckpt when resume=True ----
            state = TrainingState()
            skip_training = False
            if resume:
                loaded = _try_load_checkpoint(model_file, device, logger,
                                              model, optimizer, scheduler=scheduler)
                if loaded is not None:
                    state = loaded
                    if getattr(state, '_done_training', False):
                        logger.log(
                            f'[resume] training already finished at epoch {state.start_epoch - 1} '
                            f'(best epoch={state.best_epoch}, best_val_MSE='
                            f'{state.best_val_mse:.4f}); going to test phase.'
                        )
                        skip_training = True
                    else:
                        logger.log(
                            f'[resume] continuing from epoch {state.start_epoch} '
                            f'(best epoch={state.best_epoch}, best_val_MSE='
                            f'{state.best_val_mse:.4f}, no_improve={state.no_improve}/{patience}, '
                            f'completed_so_far={len(state.epoch_times)} epochs).'
                        )

            # ---- Per-epoch summary header ----
            if not skip_training:
                logger.subsection(
                    'Per-epoch summary  '
                    '(train + val metrics, ETA = est. time to early-stop or max-epoch)'
                )
                header = (
                    f'  {"ep":>3} | '
                    f'{"tr_loss":>8} {"tr_RMSE":>8} {"tr_r":>6} | '
                    f'{"v_RMSE":>7} {"v_MAE":>7} {"v_SD":>7} {"v_r":>6} | '
                    f'{"best_vR":>7} | '
                    f'{"lr":>9} | '
                    f'{"ep_time":>8} | {"ETA":>9} | '
                    f'{"best":>4} {"pat":>5}'
                )
                logger.log(header, with_ts=False)

                # ---- Training loop ----
                early_stopped = False
                last_epoch = state.start_epoch - 1
                for epoch in range(state.start_epoch, num_epochs + 1):
                    ep_t0 = time.time()
                    # LR actually used this epoch (read before scheduler.step)
                    current_lr = float(optimizer.param_groups[0]['lr'])
                    current_lr_esm = (
                        float(optimizer.param_groups[1]['lr'])
                        if len(optimizer.param_groups) > 1 else None
                    )
                    tr_loss, tr_rmse, tr_r, _ = train_one_epoch(
                        model, device, train_loader, optimizer, loss_fn,
                        epoch, log_interval, logger,
                        amp_dtype=amp_dtype, grad_scaler=grad_scaler,
                        trainable_params=trainable_params,
                        aux_loss_weight=aux_loss_weight,
                    )
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    G, P = predict(model, device, valid_loader, amp_dtype=amp_dtype)
                    v_rmse, v_mae, v_r, v_sd = metric.evaluate(G, P)
                    val_mse = float(mse(G, P))

                    improved = val_mse < state.best_val_mse
                    if improved:
                        state.best_val_mse = val_mse
                        state.best_epoch = epoch
                        # Cast to Python float: metric.evaluate returns numpy.float32/64,
                        # storing in tuple breaks json.dump(history_file) with
                        # "Object of type float32 is not JSON serializable".
                        state.best_val_metrics = (
                            float(v_rmse), float(v_mae), float(v_sd), float(v_r)
                        )
                        state.no_improve = 0
                        state.best_state_dict = _extract_savable_state(model)
                        flag = '*'
                    else:
                        state.no_improve += 1
                        flag = ' '

                    ep_time = time.time() - ep_t0
                    state.epoch_times.append(ep_time)

                    avg_ep = float(np.mean(state.epoch_times[-5:]))
                    remain_by_max = num_epochs - epoch
                    remain_by_patience = max(patience - state.no_improve, 0)
                    remain = max(min(remain_by_max, remain_by_patience), 0)
                    eta_sec = avg_ep * remain
                    best_val_rmse = (state.best_val_metrics[0]
                                     if state.best_val_metrics is not None else float('inf'))

                    # Append history (for plotting later)
                    state.history.append({
                        'epoch': int(epoch),
                        'tr_loss': float(tr_loss),
                        'tr_rmse': float(tr_rmse),
                        'tr_r': float(tr_r),
                        'v_rmse': float(v_rmse),
                        'v_mae': float(v_mae),
                        'v_sd': float(v_sd),
                        'v_r': float(v_r),
                        'val_mse': float(val_mse),
                        'lr': current_lr,
                        'lr_esm': current_lr_esm,
                        'ep_time_sec': float(ep_time),
                        'improved': bool(improved),
                    })

                    logger.log(
                        f'  {epoch:>3} | '
                        f'{tr_loss:>8.4f} {tr_rmse:>8.4f} {tr_r:>6.3f} | '
                        f'{v_rmse:>7.4f} {v_mae:>7.4f} {v_sd:>7.4f} {v_r:>6.3f} | '
                        f'{best_val_rmse:>7.4f} | '
                        f'{current_lr:>9.2e} | '
                        f'{fmt_secs(ep_time):>8} | {fmt_secs(eta_sec):>9} | '
                        f'{state.best_epoch:>3}{flag} {state.no_improve:>2}/{patience:<2}',
                        with_ts=False,
                    )

                    # ---- LR scheduler step (may change lr for next epoch) ----
                    if scheduler is not None:
                        prev_lr = current_lr
                        prev_lr_esm = current_lr_esm
                        if sched_step_kind == 'plateau':
                            scheduler.step(val_mse)
                        else:
                            scheduler.step()
                        new_lr = float(optimizer.param_groups[0]['lr'])
                        new_lr_esm = (
                            float(optimizer.param_groups[1]['lr'])
                            if len(optimizer.param_groups) > 1 else None
                        )
                        if abs(new_lr - prev_lr) > 1e-12:
                            msg = f'  [lr] scheduler reduced base lr: {prev_lr:.2e} -> {new_lr:.2e}'
                            if prev_lr_esm is not None and new_lr_esm is not None \
                                    and abs(new_lr_esm - prev_lr_esm) > 1e-12:
                                msg += f'  | esm lr: {prev_lr_esm:.2e} -> {new_lr_esm:.2e}'
                            logger.log(msg)

                    last_epoch = epoch
                    early_stopped = state.no_improve >= patience
                    is_last_epoch = early_stopped or epoch >= num_epochs

                    # Save full ckpt every epoch for resume after Ctrl+C
                    _save_checkpoint(
                        model_file, model=model, optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch, state=state, done_training=is_last_epoch,
                    )

                    if early_stopped:
                        logger.log(f'>>> Early stop at epoch {epoch} '
                                   f'(no improvement for {patience} epochs)')
                        break

                if not early_stopped and last_epoch >= num_epochs:
                    logger.log(f'>>> Reached max_epochs={num_epochs}')

            # ---- Best ckpt summary ----
            logger.subsection('Best checkpoint (selected by validation MSE)')
            logger.log(f'  best_epoch    : {state.best_epoch}')
            logger.log(f'  best_val_MSE  : {state.best_val_mse:.4f}')
            if state.best_val_metrics is not None:
                br, bm, bs, bp = state.best_val_metrics
                logger.log(f'  val_RMSE      : {br:.4f}')
                logger.log(f'  val_MAE       : {bm:.4f}')
                logger.log(f'  val_SD        : {bs:.4f}')
                logger.log(f'  val_Pearson_r : {bp:.4f}')
            logger.log(f'  saved weights : {model_file}')

            # ---- Save history JSON separately for plotting (no ckpt deserialization) ----
            history_file = os.path.join(
                save_root,
                f'{model_name}_{split.name}_{exp_tag}_{repeat}_history.json')
            try:
                payload = _to_jsonable({
                    'model': model_name,
                    'split': split.name,
                    'exp_tag': exp_tag,
                    'repeat': repeat,
                    'seed': seed,
                    'best_epoch': state.best_epoch,
                    'best_val_mse': state.best_val_mse,
                    'best_val_metrics': state.best_val_metrics,
                    'config': config_path,
                    'history': state.history,
                })
                with open(history_file, 'w') as f:
                    json.dump(payload, f, indent=2)
                logger.log(f'  history json  : {history_file}')
            except Exception as e:
                logger.log(f'  [warn] failed to save history json: {e}')

            # ---- Load best weights into model (for test) ----
            if state.best_state_dict is not None:
                missing, unexpected = model.load_state_dict(
                    state.best_state_dict, strict=False)
                non_backbone_missing = [k for k in missing if not _is_backbone_key(k)]
                if non_backbone_missing:
                    logger.log(f'  [load][warn] missing non-backbone keys: {non_backbone_missing[:5]} ...')
                if unexpected:
                    logger.log(f'  [load][warn] unexpected keys: {len(unexpected)} (ignored)')
            else:
                logger.log('  [warn] no best_state_dict in memory; using current model state.')

            # ---- Test: only test_tags missing this repeat in CSV ----
            logger.subsection('Final test on external sets '
                              '(using the best-val checkpoint above)')
            for tag, loader in test_loaders.items():
                if resume and (repeat in already_done.get(tag, set())):
                    logger.log(f'  [resume] {tag}: repeat {repeat} already in CSV, skipping test.')
                    continue
                if len(loader.dataset) == 0:
                    logger.log(f'  [skip] {tag}: empty test set')
                    continue
                G, P = predict(model, device, loader, amp_dtype=amp_dtype)
                rmse, mae, sd, r = evaluate_and_log(metric, G, P, tag, logger)
                with open(os.path.join(save_root,
                          f'pred_{tag.lower()}_{model_name}_{split.name}_{exp_tag}_{repeat}.json'),
                          'w') as f:
                    json.dump({'pred': P.tolist(), 'ground_truth': G.tolist()}, f)
                with open(result_files[tag], 'a+') as f:
                    # Defensive: if file lacks trailing '\n' (e.g. manual edit),
                    # append newline before new row to avoid concatenation.
                    f.seek(0, os.SEEK_END)
                    pos = f.tell()
                    if pos > 0:
                        f.seek(pos - 1)
                        if f.read(1) != '\n':
                            f.write('\n')
                    f.write(','.join([
                        str(repeat),
                        f'{rmse:.4f}', f'{mae:.4f}',
                        f'{sd:.4f}',   f'{r:.4f}',
                    ]) + '\n')
                # Mark tag complete after writing row to avoid mis-detection later
                already_done.setdefault(tag, set()).add(repeat)

            logger.log(f'>>> repeat {repeat + 1}/{n_repeats} done. '
                       f'elapsed since start: {fmt_secs(time.time() - overall_start)}')

            time.sleep(2)

        # ---- Aggregate over repeats (skip if already aggregated) ----
        for tag, rf in result_files.items():
            if resume and (-1 in already_done.get(tag, set())):
                logger.log(f'[resume] {rf} already aggregated, skipping cal_final_results.')
                continue
            try:
                cal_final_results(result_file=rf, n_repeat=n_repeats, logger=logger)
            except Exception as e:
                logger.log(f'[warn] cal_final_results failed for {rf}: {e}')

    logger.section(f'ALL DONE  |  total elapsed: {fmt_secs(time.time() - overall_start)}',
                   char='#')
    logger.close()

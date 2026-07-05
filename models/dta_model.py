"""Top-level assembly: LigandEncoder + ProteinEncoder + (optional struct tower) + Fusion -> pK."""
from __future__ import annotations

import os
from typing import List, Optional

import torch
import torch.nn as nn

from .fusion import build_fusion
from .ligand_chemberta import ChemBERTaLigandEncoder
from .protein_esm import ProteinESMEncoder
from .protein_prott5 import ProtT5ProteinEncoder
from .struct_tower import ShellGraphStructEncoder


class DTAModel(nn.Module):
    def __init__(self, ligand_encoder: nn.Module, protein_encoder: nn.Module,
                 fusion: nn.Module, plitext_encoder: Optional[nn.Module] = None):
        super().__init__()
        self.lig = ligand_encoder
        self.prot = protein_encoder
        self.fusion = fusion
        self.plitext = plitext_encoder
        self.lig_uses_smiles = bool(getattr(ligand_encoder, 'expects_smiles', False))

    def forward(self, pyg_batch, seqs: List[str],
                smiles: Optional[List[str]] = None,
                pdb_codes: Optional[List[str]] = None) -> torch.Tensor:
        if self.lig_uses_smiles:
            assert smiles is not None
            lig_out = self.lig(smiles)
        else:
            lig_out = self.lig(pyg_batch)
        if isinstance(lig_out, tuple):
            h_lig, lig_mask = lig_out
        else:
            h_lig, lig_mask = lig_out, None
        prot_out = self.prot(seqs)
        if isinstance(prot_out, tuple):
            h_prot, prot_mask = prot_out
        else:
            h_prot, prot_mask = prot_out, None
        dev = h_lig.device
        if self.plitext is not None:
            assert pdb_codes is not None
            h_pli = self.plitext(pdb_codes, device=dev)
            return self.fusion(h_lig, h_prot, h_pli, prot_mask, lig_mask=lig_mask)
        return self.fusion(h_lig, h_prot, prot_mask, lig_mask=lig_mask)

    def esm_param_names(self):
        esm = getattr(self.prot, 'esm', None)
        if esm is None:
            return []
        return [n for n, _ in esm.named_parameters()]


def _abs_path_from_root(p):
    if p and not os.path.isabs(p):
        _here = os.path.dirname(os.path.abspath(__file__))
        _root = os.path.abspath(os.path.join(_here, os.pardir))
        return os.path.join(_root, p)
    return p


def _build_protein_esm(prot_cfg: dict, exp_cfg: dict) -> ProteinESMEncoder:
    lora_cfg = prot_cfg.get('lora', {}) or {}
    adapter_cfg = prot_cfg.get('adapter', {}) or {}
    return ProteinESMEncoder(
        model_name=prot_cfg.get('model_name', 'facebook/esm2_t33_650M_UR50D'),
        mode=exp_cfg.get('esm_mode', 'frozen').split('_')[0],
        pool=prot_cfg.get('pool', 'mean'),
        max_len=int(prot_cfg.get('max_len', 1022)),
        out_dim=int(prot_cfg.get('out_dim', 128)),
        dtype=prot_cfg.get('dtype', 'bfloat16'),
        hf_cache_dir=prot_cfg.get('hf_cache_dir'),
        gradient_checkpointing=bool(prot_cfg.get('gradient_checkpointing', False)),
        lora_r=int(lora_cfg.get('r', 0) or 0),
        lora_alpha=int(lora_cfg.get('alpha', 16) or 16),
        lora_dropout=float(lora_cfg.get('dropout', 0.05) or 0.05),
        lora_target_modules=list(lora_cfg.get('target_modules', []) or []),
        lora_layers=(list(lora_cfg.get('layers', []) or []) or None),
        adapter_cfg=adapter_cfg if adapter_cfg.get('enabled') else None,
        cache_path=_abs_path_from_root(prot_cfg.get('cache_path') or None),
        cache_in_memory=bool(prot_cfg.get('cache_in_memory', True)),
        partial_cache_path=_abs_path_from_root(prot_cfg.get('partial_cache_path') or None),
        proj_cfg=prot_cfg.get('proj'),
    )


def _build_ligand_chemberta(lig_cfg: dict) -> ChemBERTaLigandEncoder:
    lora_cfg = lig_cfg.get('lora', {}) or {}
    adapter_cfg = lig_cfg.get('adapter', {}) or {}
    return ChemBERTaLigandEncoder(
        model_name=lig_cfg.get('model_name', 'DeepChem/ChemBERTa-77M-MLM'),
        mode=lig_cfg.get('mode', 'frozen'),
        pool=lig_cfg.get('pool', 'cls'),
        max_len=int(lig_cfg.get('max_len', 512)),
        out_dim=int(lig_cfg.get('out_dim', 128)),
        dtype=lig_cfg.get('dtype', 'bfloat16'),
        hf_cache_dir=lig_cfg.get('hf_cache_dir'),
        gradient_checkpointing=bool(lig_cfg.get('gradient_checkpointing', False)),
        lora_r=int(lora_cfg.get('r', 0) or 0),
        lora_alpha=int(lora_cfg.get('alpha', 16) or 16),
        lora_dropout=float(lora_cfg.get('dropout', 0.05) or 0.05),
        lora_target_modules=list(lora_cfg.get('target_modules', []) or []),
        lora_layers=(list(lora_cfg.get('layers', []) or []) or None),
        adapter_cfg=adapter_cfg if adapter_cfg.get('enabled') else None,
        cache_path=_abs_path_from_root(lig_cfg.get('cache_path') or None),
        cache_in_memory=bool(lig_cfg.get('cache_in_memory', True)),
        partial_cache_path=_abs_path_from_root(lig_cfg.get('partial_cache_path') or None),
        proj_cfg=lig_cfg.get('proj_cfg') or lig_cfg.get('proj'),
    )


def _build_protein_prott5(prot_cfg: dict) -> ProtT5ProteinEncoder:
    return ProtT5ProteinEncoder(
        model_name=prot_cfg.get('model_name', 'Rostlab/prot_t5_xl_bfd'),
        pool=prot_cfg.get('pool', 'mean'),
        max_len=int(prot_cfg.get('max_len', 512)),
        out_dim=int(prot_cfg.get('out_dim', 1024)),
        dtype=prot_cfg.get('dtype', 'bfloat16'),
        hf_cache_dir=prot_cfg.get('hf_cache_dir'),
        cache_path=_abs_path_from_root(prot_cfg.get('cache_path') or None),
        cache_in_memory=bool(prot_cfg.get('cache_in_memory', True)),
        proj_cfg=prot_cfg.get('proj_cfg') or prot_cfg.get('proj'),
    )


def _build_struct_tower(pli_cfg: dict) -> ShellGraphStructEncoder:
    return ShellGraphStructEncoder(
        features_path=_abs_path_from_root(
            pli_cfg.get('features_path') or 'features_residue/residue_N60.npz'),
        scaler_path=_abs_path_from_root(
            pli_cfg.get('scaler_path') or 'features_residue/residue_N60_scaler.npz'),
        out_dim=int(pli_cfg.get('out_dim', 128)),
        n_pairs=int(pli_cfg.get('n_pairs', 168)),
        n_shells=int(pli_cfg.get('n_shells', 60)),
        gnn_type=str(pli_cfg.get('gnn_type', 'GAT_GCN')),
        hidden_dim=int(pli_cfg.get('hidden_dim', 128)),
        n_layers=int(pli_cfg.get('n_layers', 3)),
        k_neighbors=int(pli_cfg.get('k_neighbors', 2)),
        n_heads=int(pli_cfg.get('n_heads', 4)),
        lstm_num_layers=int(pli_cfg.get('lstm_num_layers', 1)),
        use_ffn=bool(pli_cfg.get('use_ffn', True)),
        pos_encoding=str(pli_cfg.get('pos_encoding', 'none')),
        pooling=str(pli_cfg.get('pooling', 'flatten')),
        dropout=float(pli_cfg.get('dropout', 0.1)),
        proj_dropout=float(pli_cfg.get('proj_dropout', 0.2)),
        missing_policy=str(pli_cfg.get('missing_policy', 'zero')),
        train_mode=str(pli_cfg.get('train_mode', 'e2e')),
        tf_ffn_mult=int(pli_cfg.get('tf_ffn_mult', 2)),
        tf_dropout=float(pli_cfg.get('tf_dropout', -1.0)),
    )


def setup_struct_branch(model: DTAModel, pli_cfg: dict, split: str, repeat: int) -> None:
    """Apply struct-tower train_mode / optional pretrained weights."""
    if model.plitext is None or not isinstance(model.plitext, ShellGraphStructEncoder):
        return
    enc = model.plitext
    mode = str(pli_cfg.get('train_mode', 'e2e')).lower()
    enc.train_mode = mode
    if mode == 'pretrained_init':
        tpl = pli_cfg.get('pretrained_ckpt_template')
        if tpl:
            ckpt = tpl.format(split=split, fold=repeat, repeat=repeat)
            enc.load_pretrained_backbone(ckpt)
        else:
            enc.apply_train_mode()
    else:
        enc.apply_train_mode()


def build_model(config: dict, ligand_model_name: str) -> DTAModel:
    del ligand_model_name  # used only for result file naming in train_runner
    lig_cfg = config.get('ligand', {})
    prot_cfg = config.get('protein', {})
    fus_cfg = config.get('fusion', {})
    exp_cfg = config.get('experiment', {})
    out_dim = int(lig_cfg.get('out_dim', 128))

    lig_type = str(lig_cfg.get('type', 'chemberta')).lower()
    if lig_type == 'chemberta':
        lig_enc = _build_ligand_chemberta(lig_cfg)
    else:
        raise ValueError(f"Unknown ligand.type={lig_type!r}; TriFusion supports 'chemberta' only.")

    prot_type = str(prot_cfg.get('type', 'esm2')).lower()
    if prot_type == 'esm2':
        prot_enc = _build_protein_esm(prot_cfg, exp_cfg)
    elif prot_type == 'prott5':
        prot_enc = _build_protein_prott5(prot_cfg)
    else:
        raise ValueError(f"Unknown protein.type={prot_type!r}; TriFusion supports 'esm2' and 'prott5'.")

    pli_cfg = config.get('plitext') or {}
    plitext_enc = None
    d_plitext = 0
    if pli_cfg.get('enabled', False):
        pli_type = str(pli_cfg.get('type', 'shell_graph')).lower()
        if pli_type in ('onion_graph', 'oniongraph', 'shell_graph'):
            plitext_enc = _build_struct_tower(pli_cfg)
        else:
            raise ValueError(f"Unknown plitext.type={pli_type!r}")
        d_plitext = int(pli_cfg.get('out_dim', 128))

    fusion = build_fusion(fus_cfg, d_lig=out_dim,
                          d_prot=int(prot_cfg.get('out_dim', 128)),
                          d_plitext=d_plitext)
    return DTAModel(lig_enc, prot_enc, fusion, plitext_encoder=plitext_enc)

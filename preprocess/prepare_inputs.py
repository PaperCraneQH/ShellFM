"""Locate raw protein/ligand files and convert ligands to .pdb (vendored from OnionNet-2/baseline)."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    _HAS_RDKIT = True
except Exception:  # noqa: BLE001
    _HAS_RDKIT = False

OBABEL_BIN = shutil.which('obabel')


@dataclass
class DataRoots:
    """Root directories of the raw structural data (configured per user machine).

    - casf_root       : CASF-2016 coreset root (one <pdb>/ subdir each)
    - pdbbind_pl_root : PDBbind Protein-Ligand root (bucketed by year underneath)
    - csar_root       : CSAR-HiQ root (one <pdb>/ subdir each)
    - pdbbind_index   : PDBbind index CSV; must contain PDB_code and file(year bucket) columns
    """
    casf_root: Optional[Path] = None
    pdbbind_pl_root: Optional[Path] = None
    csar_root: Optional[Path] = None
    pdbbind_index: Optional[Path] = None

    def __post_init__(self):
        for attr in ('casf_root', 'pdbbind_pl_root', 'csar_root', 'pdbbind_index'):
            v = getattr(self, attr)
            if v is not None:
                setattr(self, attr, Path(v))


def build_pdbbind_year_map(pdbbind_index: Optional[Path]) -> Dict[str, str]:
    """Return {pdb_code(lower): year_bucket}; empty map if the index or its file column is missing."""
    if pdbbind_index is None or not Path(pdbbind_index).is_file():
        return {}
    import pandas as pd
    df = pd.read_csv(pdbbind_index)
    if 'file' not in df.columns or 'PDB_code' not in df.columns:
        return {}
    return dict(zip(df['PDB_code'].astype(str).str.lower(), df['file'].astype(str)))


def find_protein_ligand(pdb: str, year_map: Dict[str, str],
                        roots: DataRoots) -> Tuple[Optional[Path], Optional[Path], Optional[str]]:
    """Return (protein_pdb_path, ligand_path, source_tag); (None, None, None) if not found."""
    pdb = pdb.lower()

    if roots.casf_root is not None:
        prot = roots.casf_root / pdb / f'{pdb}_protein.pdb'
        if prot.exists():
            lig = roots.casf_root / pdb / f'{pdb}_ligand.sdf'
            if not lig.exists():
                lig = roots.casf_root / pdb / f'{pdb}_ligand.mol2'
            return prot, lig, 'CASF-2016'

    if roots.pdbbind_pl_root is not None and pdb in year_map:
        year = year_map[pdb]
        prot = roots.pdbbind_pl_root / year / pdb / f'{pdb}_protein.pdb'
        lig = roots.pdbbind_pl_root / year / pdb / f'{pdb}_ligand.sdf'
        if not lig.exists():
            lig = roots.pdbbind_pl_root / year / pdb / f'{pdb}_ligand.mol2'
        if prot.exists():
            return prot, lig, f'PDBbind/{year}'

    if roots.csar_root is not None:
        prot = roots.csar_root / pdb / f'{pdb}_protein.pdb'
        if prot.exists():
            lig = roots.csar_root / pdb / f'{pdb}_ligand.mol2'
            return prot, lig, 'CSAR-HiQ'

    return None, None, None


def _convert_lig_rdkit(lig_in: Path, out_pdb: Path) -> bool:
    if not _HAS_RDKIT:
        return False
    suffix = lig_in.suffix.lower()
    mols = []
    for sanitize in (True, False):
        try:
            if suffix == '.sdf':
                mols = [m for m in Chem.SDMolSupplier(str(lig_in), removeHs=False,
                                                      sanitize=sanitize) if m is not None]
            elif suffix == '.mol2':
                m = Chem.MolFromMol2File(str(lig_in), removeHs=False, sanitize=sanitize)
                mols = [m] if m is not None else []
            else:
                return False
        except Exception:  # noqa: BLE001
            mols = []
        if mols:
            break
    if not mols:
        return False
    try:
        Chem.MolToPDBFile(mols[0], str(out_pdb))
    except Exception:  # noqa: BLE001
        return False
    return out_pdb.exists() and out_pdb.stat().st_size > 0


def _convert_lig_obabel(lig_in: Path, out_pdb: Path) -> bool:
    if not OBABEL_BIN or not Path(OBABEL_BIN).exists():
        return False
    try:
        proc = subprocess.run(
            [OBABEL_BIN, str(lig_in), '-O', str(out_pdb)],
            capture_output=True, text=True, timeout=60,
        )
    except Exception:  # noqa: BLE001
        return False
    return proc.returncode == 0 and out_pdb.exists() and out_pdb.stat().st_size > 0


def convert_lig(lig_in: Path, out_pdb: Path) -> Tuple[bool, str]:
    """Try RDKit first, then fall back to obabel. Returns (ok, tool_used)."""
    if _convert_lig_rdkit(lig_in, out_pdb):
        return True, 'rdkit'
    if _convert_lig_obabel(lig_in, out_pdb):
        return True, 'obabel'
    return False, ''

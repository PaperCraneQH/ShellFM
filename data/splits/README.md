# PDBbind v2020 Data Splits

This directory ships the **metadata and split CSVs** used by ShellFM. Raw protein–ligand structures (PDB/mol2/sdf files) are **not** included; obtain them from [PDBbind](http://www.pdbbind.org.cn/), [CASF-2016](http://www.pdbbind.org.cn/casf.php), and [CSAR-HiQ](http://www.csardock.org).

## Layout

```
splits/
├── PL-2020R1.csv                 # index: PDB_code, file (year bucket)
├── PL-2020R1_seq_smiles.csv      # PDB_code, sequence, smiles, -logKd/Ki
├── PL-2020R1_train.csv           # training pool (PDBbind v2020 minus CASF-2016)
├── CASF-2016.csv                 # external benchmark (285 complexes)
├── CSAR-HiQ_seq_smiles.csv       # full CSAR-HiQ benchmark (343 complexes)
├── CSAR-HiQ_dedup.csv            # CSAR eval subset (81 complexes, no train overlap)
└── cold_start/                   # five-fold split CSVs
    ├── base/       train_base_{0..4}.csv, valid_base_{0..4}.csv
    ├── random/     train/valid/test_random_{0..4}.csv
    ├── scaffold/   train/valid/test_scaffold_{0..4}.csv
    ├── seq_identity/ train/valid/test_seq_identity_{0..4}.csv
    └── holdout/    train.csv, valid_holdout_2018.csv, test_holdout_2019.csv
```

## External benchmarks

| Benchmark | CSV | Size | Used for evaluation? |
|-----------|-----|------|--------------------|
| **CASF-2016** | `CASF-2016.csv` | 285 | Yes (full set) |
| **CSAR-HiQ (full)** | `CSAR-HiQ_seq_smiles.csv` | 343 | **No** — 262 complexes overlap `PL-2020R1_train.csv` |
| **CSAR-HiQ (eval)** | `CSAR-HiQ_dedup.csv` | **81** | **Yes** — zero overlap with the training pool |

### CSAR-HiQ deduplication and data leakage

CSAR-HiQ ships 343 protein–ligand complexes. Because PDBbind v2020 and CSAR-HiQ share many structures, **262 of the 343 CSAR complexes also appear in the training pool** (`PL-2020R1_train.csv`). Reporting metrics on all 343 complexes would inflate performance due to **train–test leakage**.

ShellFM therefore evaluates CSAR-HiQ on the **81-complex deduplicated subset** (`CSAR-HiQ_dedup.csv`), obtained by removing every CSAR complex whose `PDB_code` is present in `PL-2020R1_train.csv`. Sequence and SMILES labels are joined from `CSAR-HiQ_seq_smiles.csv` when building `external/CSAR-HiQ.pt` (see `preprocess/3_build_pyg_dataset.py`).

When fitting the feature scaler (`preprocess/2_fit_scaler.py`), exclude both external benchmarks from the statistics pool so that neither CASF nor CSAR complexes influence normalization.

## Evaluation protocol

| Split | Train / Valid | Test |
|-------|---------------|------|
| **base** | PDBbind v2020 (minus CASF-2016), 9:1 | CASF-2016 (285) + CSAR-HiQ (**81**, dedup) |
| **random** | random 5-fold | in-fold test |
| **scaffold** | Bemis–Murcko scaffold split | in-fold test |
| **seq_identity** | sequence-identity split | in-fold test |
| **holdout** | pre-2018 train | 2018 valid / 2019 test (temporal) |

Each cold-start split uses **5 repeats** (`_0` … `_4`). The `base` split has no in-fold test set; external benchmarks serve as test.

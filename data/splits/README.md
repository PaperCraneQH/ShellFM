# PDBbind v2020 Data Splits

This directory ships the **metadata and split CSVs** used by ShellLM. Raw protein–ligand structures (PDB/mol2/sdf files) are **not** included; obtain them from [PDBbind](http://www.pdbbind.org.cn/), [CASF-2016](http://www.pdbbind.org.cn/casf.php), and [CSAR-HiQ](http://www.csardock.org).

## Layout

```
splits/
├── PL-2020R1.csv                 # index: PDB_code, file (year bucket)
├── PL-2020R1_seq_smiles.csv      # PDB_code, sequence, smiles, -logKd/Ki
├── PL-2020R1_train.csv           # training-pool PDB codes (CSAR deduplication)
├── CASF-2016.csv                 # external benchmark labels
├── CSAR-HiQ_seq_smiles.csv       # external benchmark (sequence + SMILES)
└── cold_start/                   # five-fold split CSVs
    ├── base/       train_base_{0..4}.csv, valid_base_{0..4}.csv
    ├── random/     train/valid/test_random_{0..4}.csv
    ├── scaffold/   train/valid/test_scaffold_{0..4}.csv
    ├── seq_identity/ train/valid/test_seq_identity_{0..4}.csv
    └── holdout/    train.csv, valid_holdout_2018.csv, test_holdout_2019.csv
```

## Evaluation protocol

| Split | Train / Valid | Test |
|-------|---------------|------|
| **base** | PDBbind v2020 (minus CASF-2016), 9:1 | CASF-2016 + CSAR-HiQ (external) |
| **random** | random 5-fold | in-fold test |
| **scaffold** | Bemis–Murcko scaffold split | in-fold test |
| **seq_identity** | sequence-identity split | in-fold test |
| **holdout** | pre-2018 train | 2018 valid / 2019 test (temporal) |

Each cold-start split uses **5 repeats** (`_0` … `_4`). The `base` split has no in-fold test set; external benchmarks serve as test.

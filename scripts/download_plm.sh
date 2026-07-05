#!/usr/bin/env bash
# Prefetch TriFusion PLM weights into the HuggingFace cache.
# Training downloads models automatically; this script is for offline prefetch.
#
# Usage:
#   bash scripts/download_plm.sh              # default cache (~/.cache/huggingface)
#   HF_HOME=/data/hf_cache bash scripts/download_plm.sh
#   bash scripts/download_plm.sh efficient    # ESM2-t6 + ChemBERTa only
set -euo pipefail

MODE="${1:-default}"
# export HF_ENDPOINT="https://hf-mirror.com"   # uncomment for mirror

echo "[download] HF cache = ${HF_HOME:-$HOME/.cache/huggingface}"

dl () {
  echo "[download] $1 ..."
  huggingface-cli download "$1" --resume-download >/dev/null
  echo "[download] done: $1"
}

dl "DeepChem/ChemBERTa-77M-MTR"

if [ "$MODE" = "efficient" ]; then
  dl "facebook/esm2_t6_8M_UR50D"
else
  dl "Rostlab/prot_t5_xl_half_uniref50-enc"
fi

echo "[download] Done."

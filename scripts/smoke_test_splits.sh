#!/usr/bin/env bash
# Validate the TriFusion training pipeline on all five splits (1 epoch, 1 repeat).
# Requires: activated trifusion conda env, preprocessed data, and HuggingFace model cache.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Warning: CONDA_PREFIX is unset. Activate the trifusion env first:" >&2
  echo "  conda activate trifusion" >&2
fi

export LD_LIBRARY_PATH="${CONDA_PREFIX:+$CONDA_PREFIX/lib:}${LD_LIBRARY_PATH:-}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
export HF_HOME="${HF_HOME:-$HUGGINGFACE_HUB_CACHE}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

PY="${PYTHON:-python}"
CFG="${CONFIG:-configs/trifusion_smoke_esm2.yaml}"

for script in train_base train_random train_scaffold train_seq_identity train_holdout; do
  echo "========== $script =========="
  "$PY" "training/${script}.py" \
    --config "$CFG" \
    --data_root data_processed_esm \
    --device "${DEVICE:-cuda:0}" \
    --no_resume
done

echo "All five splits completed successfully."

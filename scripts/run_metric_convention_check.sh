#!/usr/bin/env bash
# Run the macro-vs-micro metric check on the AutoDL server.
# Read-only with respect to training: no training starts, no checkpoint is
# written, nothing is selected on the test split.
set -Eeuo pipefail

REPO=/root/autodl-tmp/BetterLViT
PY=/root/autodl-tmp/envs/betterlvit/bin/python3.11
CKPT="${1:-}"

if [[ -z "$CKPT" ]]; then
  echo "usage: $0 <path-to-V4B-best_model-BetterLViT.pth.tar>" >&2
  echo "hint:  ls -d $REPO/Covid19/BetterLViT/*/models/best_model-BetterLViT.pth.tar" >&2
  exit 2
fi

cd "$REPO"
test -f "$CKPT"
test -x "$PY"

# Refuse to run while a training process tree is alive (GPU contention).
if ps -eo comm=,args= | awk '$1 ~ /^python/ && $0 ~ /train_model\.py/ {found=1} END{exit !found}'; then
  echo "A train_model.py process is running. Refusing to share the GPU." >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export BETTERLVIT_VIS_FREQUENCY=0
# Keep Config.py's validators happy without changing any experiment setting.
export BETTERLVIT_TEXT_MODALITY_DROPOUT_PROB=0.0
export BETTERLVIT_SPLIT_PROTOCOL=legacy
unset BETTERLVIT_RESUME_PATH || true
unset BETTERLVIT_RESUME_SHA256 || true

"$PY" tools/evaluate_metric_conventions.py \
  --checkpoint "$CKPT" \
  --output "$REPO/metric_convention_report.json"

echo
echo "Report: $REPO/metric_convention_report.json"

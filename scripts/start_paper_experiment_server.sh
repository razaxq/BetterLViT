#!/usr/bin/env bash
set -euo pipefail

experiment="${1:-b0_baseline}"
seed="${2:-1219}"
epochs="${3:-100}"
batch_size="${4:-16}"

case "$experiment" in
  b0_baseline|a0_lora|a1_lora_focal|a2_lora_freq|a3_lora_fmiseg) ;;
  *) echo "Unsupported experiment: $experiment" >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${BETTERLVIT_PYTHON:-/root/autodl-tmp/envs/betterlvit-paper/bin/python}"
runtime_dir="$repo_root/runtime_logs"
lock_file="$runtime_dir/training.lock"
mkdir -p "$runtime_dir"

if pgrep -af '[t]rain_model.py' >/dev/null; then
  echo 'A train_model.py process is already active; refusing a duplicate.' >&2
  pgrep -af '[t]rain_model.py' >&2
  exit 3
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
stdout="$runtime_dir/train_${experiment}_${timestamp}.stdout.log"
stderr="$runtime_dir/train_${experiment}_${timestamp}.stderr.log"
metadata="$runtime_dir/paper_experiment_current.env"

export BETTERLVIT_EXPERIMENT="$experiment"
export BETTERLVIT_SEED="$seed"
export BETTERLVIT_EPOCHS="$epochs"
export BETTERLVIT_BATCH_SIZE="$batch_size"
export BETTERLVIT_TRAIN_DROP_LAST=1
export BETTERLVIT_NUM_WORKERS=4
export BETTERLVIT_DETERMINISTIC=1
export BETTERLVIT_CUDNN_ENABLED=1
export BETTERLVIT_VIS_FREQUENCY=100000
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED="$seed"
export BETTERLVIT_GIT_COMMIT="$(git -C "$repo_root" rev-parse HEAD)"
export HF_HOME="${BETTERLVIT_HF_HOME:-/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_MODULES_CACHE="$HF_HOME/modules"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

nohup bash -c '
  cd "$1"
  exec flock -n "$2" "$3" train_model.py
' _ "$repo_root" "$lock_file" "$python_bin" \
  >"$stdout" 2>"$stderr" </dev/null &
launcher_pid=$!

sleep 3
if ! kill -0 "$launcher_pid" 2>/dev/null; then
  echo 'Training process exited during startup.' >&2
  tail -80 "$stdout" "$stderr" >&2 || true
  exit 5
fi

cat >"$metadata" <<EOF
EXPERIMENT=$experiment
SEED=$seed
EPOCHS=$epochs
BATCH_SIZE=$batch_size
GIT_COMMIT=$BETTERLVIT_GIT_COMMIT
DETERMINISTIC=1
CUDNN_ENABLED=1
TRAIN_DROP_LAST=1
LAUNCHER_PID=$launcher_pid
STARTED_AT=$(date -Is)
REPO=$repo_root
STDOUT=$stdout
STDERR=$stderr
HF_HOME=$HF_HOME
EOF

echo "Started $experiment (PID $launcher_pid)"
echo "Metadata: $metadata"
echo "Stdout: $stdout"
echo "Stderr: $stderr"

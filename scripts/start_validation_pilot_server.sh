#!/usr/bin/env bash
set -euo pipefail

experiment="${1:-p2_tcsrv22_single_hop_boundary}"
seed="${2:-1219}"
epochs="${3:-40}"
batch_size="${4:-16}"

case "$experiment" in
  p1_tcsrv21_boundary_router|p2_tcsrv22_single_hop_boundary|\
  p3_tcsrv23_calibrated_gate|p4_tcsrv24_sparse_boundary|\
  c0_frozen_freq_tversky|p5_tcsrv25_local_tversky)
    ;;
  *)
    echo "Unsupported validation pilot: $experiment" >&2
    exit 2
    ;;
esac
if ! [[ "$epochs" =~ ^[1-9][0-9]*$ ]] || [ "$epochs" -gt 80 ]; then
  echo "Pilot epochs must be an integer from 1 to 80." >&2
  exit 2
fi
if ! [[ "$batch_size" =~ ^[1-9][0-9]*$ ]]; then
  echo "Batch size must be a positive integer." >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${BETTERLVIT_PYTHON:-/root/autodl-tmp/envs/betterlvit-paper/bin/python}"
runtime_dir="$repo_root/runtime_logs"
lock_file="$runtime_dir/training.lock"
runner="$repo_root/scripts/run_validation_pilot_server.sh"
mkdir -p "$runtime_dir"

if [ ! -x "$runner" ]; then
  echo "Missing executable validation-pilot runner: $runner" >&2
  exit 2
fi
if pgrep -af '[t]rain_model.py|[e]valuate_experiment.py' >/dev/null; then
  echo "A training/evaluation process is already active; refusing a duplicate." >&2
  pgrep -af '[t]rain_model.py|[e]valuate_experiment.py' >&2
  exit 3
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
stdout="$runtime_dir/pilot_${experiment}_${timestamp}.stdout.log"
stderr="$runtime_dir/pilot_${experiment}_${timestamp}.stderr.log"
status_file="$runtime_dir/pilot_${experiment}_${timestamp}.status"
metadata="$runtime_dir/validation_pilot_current.env"

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

nohup "$runner" \
  "$repo_root" \
  "$lock_file" \
  "$python_bin" \
  "$status_file" \
  "$experiment" \
  >"$stdout" 2>"$stderr" </dev/null &
launcher_pid=$!

sleep 3
if ! kill -0 "$launcher_pid" 2>/dev/null; then
  echo "Validation pilot exited during startup." >&2
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
AUTO_EVALUATE=0
TEST_SPLIT_ALLOWED=0
PILOT_STAGE=validation_only_40_epoch
LAUNCHER_PID=$launcher_pid
STARTED_AT=$(date -Is)
REPO=$repo_root
STDOUT=$stdout
STDERR=$stderr
STATUS=$status_file
HF_HOME=$HF_HOME
EOF

echo "Started validation-only pilot $experiment (PID $launcher_pid)"
echo "Metadata: $metadata"
echo "Stdout: $stdout"
echo "Stderr: $stderr"
echo "Test evaluation: disabled by pilot runner"

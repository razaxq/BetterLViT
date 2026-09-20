#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 7 ]; then
  echo "Usage: $0 C1_REPO P6_REPO C1_SHA P6_SHA SEED EPOCHS BATCH" >&2
  exit 2
fi
c1_repo="$1"
p6_repo="$2"
c1_sha="$3"
p6_sha="$4"
seed="$5"
epochs="$6"
batch_size="$7"
python_bin="${BETTERLVIT_PYTHON:-/root/autodl-tmp/envs/betterlvit-paper/bin/python}"
runner="$p6_repo/scripts/run_bcdh_pair_server.sh"
runtime_dir="$p6_repo/runtime_logs"
timestamp="$(date +%Y%m%d_%H%M%S)"
output_dir="$runtime_dir/bcdh_pair_${timestamp}"
status_file="$output_dir/chain.status"
metadata="$runtime_dir/bcdh_pair_current.env"

if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
  echo "Seed must be a non-negative integer." >&2
  exit 2
fi
if ! [[ "$epochs" =~ ^[1-9][0-9]*$ ]] || [ "$epochs" -gt 80 ]; then
  echo "Validation-pair epochs must be an integer from 1 to 80." >&2
  exit 2
fi
if ! [[ "$batch_size" =~ ^[1-9][0-9]*$ ]]; then
  echo "Batch size must be a positive integer." >&2
  exit 2
fi
if [ "$epochs" -ne 40 ] || [ "$batch_size" -ne 16 ] || [ "$seed" -ne 1219 ]; then
  echo "BCDH C1/P6 pilot is locked to seed 1219, 40 epochs, batch 16." >&2
  exit 2
fi
for repo in "$c1_repo" "$p6_repo"; do
  [ -d "$repo/.git" ] || [ -f "$repo/.git" ] || {
    echo "Missing Git worktree: $repo" >&2
    exit 2
  }
  if [ -n "$(git -C "$repo" status --porcelain --untracked-files=no)" ]; then
    echo "Tracked files are dirty in $repo" >&2
    exit 2
  fi
done

if pgrep -af '[t]rain_model.py|[e]xport_validation_metrics.py' >/dev/null; then
  echo "A training/export process is active; refusing duplicate launch." >&2
  pgrep -af '[t]rain_model.py|[e]xport_validation_metrics.py' >&2
  exit 3
fi
mkdir -p "$output_dir"
nohup "$runner" "$c1_repo" "$p6_repo" "$c1_sha" "$p6_sha" \
  "$python_bin" "$seed" "$epochs" "$batch_size" "$status_file" \
  "$output_dir" >"$output_dir/chain.stdout.log" \
  2>"$output_dir/chain.stderr.log" </dev/null &
launcher_pid=$!
sleep 3
if ! kill -0 "$launcher_pid" 2>/dev/null; then
  echo "BCDH pair chain exited during startup." >&2
  tail -80 "$output_dir/chain.stdout.log" "$output_dir/chain.stderr.log" >&2 || true
  exit 5
fi
cat >"$metadata" <<EOF
CHAIN=BCDH_C1_TO_P6
C1_REPO=$c1_repo
P6_REPO=$p6_repo
C1_GIT_COMMIT=$c1_sha
P6_GIT_COMMIT=$p6_sha
SEED=$seed
EPOCHS=$epochs
BATCH_SIZE=$batch_size
AUTO_TEST_EVALUATE=0
TEST_SPLIT_ALLOWED=0
LAUNCHER_PID=$launcher_pid
STARTED_AT=$(date -Is)
STATUS=$status_file
OUTPUT_DIR=$output_dir
EOF
echo "Started BCDH C1 -> P6 validation-only chain (PID $launcher_pid)"
echo "Metadata: $metadata"
echo "Test evaluation: disabled"

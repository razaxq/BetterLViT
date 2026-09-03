#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 7 ]; then
  echo "Usage: $0 C3_REPO P8_REPO C3_SHA P8_SHA SEED EPOCHS BATCH" >&2
  exit 2
fi
c3_repo="$1"
p8_repo="$2"
c3_sha="$3"
p8_sha="$4"
seed="$5"
epochs="$6"
batch_size="$7"
python_bin="${BETTERLVIT_PYTHON:-/root/autodl-tmp/envs/betterlvit-paper/bin/python}"
runner="$p8_repo/scripts/run_race_pair_server.sh"
runtime_dir="$p8_repo/runtime_logs"
timestamp="$(date +%Y%m%d_%H%M%S)"
output_dir="$runtime_dir/race_pair_${timestamp}"
status_file="$output_dir/chain.status"
metadata="$runtime_dir/race_pair_current.env"

if [ "$seed" -ne 1219 ] || [ "$epochs" -ne 80 ] || [ "$batch_size" -ne 16 ]; then
  echo "RACE C3/P8 screen is locked to seed 1219, 80 epochs, batch 16." >&2
  exit 2
fi
for repo in "$c3_repo" "$p8_repo"; do
  [ -d "$repo/.git" ] || [ -f "$repo/.git" ] || {
    echo "Missing Git worktree: $repo" >&2
    exit 2
  }
  [ -z "$(git -C "$repo" status --porcelain --untracked-files=no)" ] || {
    echo "Tracked files are dirty in $repo" >&2
    exit 2
  }
done
if pgrep -af '[t]rain_model.py|[e]xport_validation_metrics.py' >/dev/null; then
  echo "A training/export process is active; refusing duplicate launch." >&2
  pgrep -af '[t]rain_model.py|[e]xport_validation_metrics.py' >&2
  exit 3
fi

mkdir -p "$output_dir"
nohup "$runner" "$c3_repo" "$p8_repo" "$c3_sha" "$p8_sha" \
  "$python_bin" "$seed" "$epochs" "$batch_size" "$status_file" \
  "$output_dir" >"$output_dir/chain.stdout.log" \
  2>"$output_dir/chain.stderr.log" </dev/null &
launcher_pid=$!
sleep 3
if ! kill -0 "$launcher_pid" 2>/dev/null; then
  echo "RACE pair chain exited during startup." >&2
  tail -80 "$output_dir/chain.stdout.log" "$output_dir/chain.stderr.log" >&2 || true
  exit 5
fi
cat >"$metadata" <<EOF
CHAIN=RACE_C3_TO_P8
C3_REPO=$c3_repo
P8_REPO=$p8_repo
C3_GIT_COMMIT=$c3_sha
P8_GIT_COMMIT=$p8_sha
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
echo "Started RACE C3 -> P8 validation-only chain (PID $launcher_pid)"
echo "Metadata: $metadata"
echo "Test evaluation: disabled"

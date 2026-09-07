#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

REPO=/root/autodl-tmp/BetterLViT
PY=/root/autodl-tmp/envs/betterlvit/bin/python3.11
RUNTIME=/root/autodl-tmp/runtime_logs
BRANCH=codex/v4b-repro-grouped
MANIFEST="$REPO/protocols/splits/known_patient_grouped_seed1219.json"
LOCK="$RUNTIME/gpu0_training.lock"
EXPECTED_COMMIT=${BETTERLVIT_EXPECTED_COMMIT:-}

if [[ ! "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "BETTERLVIT_EXPECTED_COMMIT must be the reviewed 40-hex commit" >&2
  exit 2
fi

cd "$REPO"
test "$(git branch --show-current)" = "$BRANCH"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
GIT_STATUS="$(git status --porcelain=v1)"
test -z "$GIT_STATUS"
test -x "$PY"
test -f "$MANIFEST"
mkdir -p "$RUNTIME"
LAUNCH_LOCK="$RUNTIME/gpu0_launch.lock"
# Serialize launchers and hold the training lock throughout every GPU gate.
# The screen supervisor blocks on the same training lock during handoff while
# this process keeps LAUNCH_LOCK, leaving no window for a second launcher.
exec 8>"$LAUNCH_LOCK"
flock -n 8
exec 9>"$LOCK"
flock -n 9

test "$(df --output=avail -k /root/autodl-tmp | tail -1)" -ge 6291456
GPU_COMPUTE_PIDS="$(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader
)"
test -z "$(printf '%s\n' "$GPU_COMPUTE_PIDS" | sed '/^[[:space:]]*$/d')"
test -z "$(
  ps -eo comm=,args= |
    awk '$1 ~ /^python/ && $0 ~ /(train_model\.py|test_model\.py|evaluate_[^ ]*\.py)/ {print}'
)"
unset BETTERLVIT_RESUME_PATH
unset BETTERLVIT_RESUME_SHA256
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=1219
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export BETTERLVIT_SEED=1219
export BETTERLVIT_GIT_COMMIT="$EXPECTED_COMMIT"
export BETTERLVIT_EPOCHS=200
export BETTERLVIT_BATCH_SIZE=16
export BETTERLVIT_NUM_WORKERS=4
export BETTERLVIT_VIS_FREQUENCY=0
export BETTERLVIT_TEXT_MODALITY_DROPOUT_PROB=0.0
export BETTERLVIT_SPLIT_PROTOCOL=known_patient_grouped_sensitivity_v1
export BETTERLVIT_SPLIT_MANIFEST="$MANIFEST"

TEMP_DIR="$(mktemp -d /root/autodl-tmp/v4b_grouped_manifest.XXXXXX)"
TEMP_MANIFEST="$TEMP_DIR/manifest.json"
HANDOFF_COMPLETE=0
cleanup() {
  if [[ -n "${TEMP_MANIFEST:-}" ]]; then
    rm -f "$TEMP_MANIFEST" "${TEMP_MANIFEST}.tmp"
  fi
  if [[ -n "${TEMP_DIR:-}" ]]; then
    rmdir "$TEMP_DIR" 2>/dev/null || true
  fi
  if [[ "${HANDOFF_COMPLETE:-0}" != 1 && -n "${SCREEN:-}" ]]; then
    screen -S "$SCREEN" -X quit >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
"$PY" tools/build_covid19_grouped_split.py \
  --dataset-root datasets/Covid19 \
  --output "$TEMP_MANIFEST" \
  --seed 1219
cmp --silent "$TEMP_MANIFEST" "$MANIFEST"

"$PY" tools/check_runtime_optimizations.py
"$PY" tools/check_fam_eppa_v4b.py
"$PY" tools/check_reproducibility_harness.py
"$PY" tools/check_v4b_repro_protocol.py

GPU_COMPUTE_PIDS="$(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader
)"
test -z "$(printf '%s\n' "$GPU_COMPUTE_PIDS" | sed '/^[[:space:]]*$/d')"

STAMP="$(date +%Y%m%d_%H%M%S)"
SLUG=v4b_repro_grouped_s1219
SESSION="Test_session_${SLUG}_${STAMP}"
SESSION_DIR="$REPO/Covid19/BetterLViT/$SESSION"
test ! -e "$SESSION_DIR"
export BETTERLVIT_SESSION_NAME="$SESSION"

SCREEN="train_${SLUG}_${STAMP}"
STDOUT="$RUNTIME/train_${SLUG}_${STAMP}.stdout.log"
STDERR="$RUNTIME/train_${SLUG}_${STAMP}.stderr.log"
HEARTBEAT="$RUNTIME/train_${SLUG}_${STAMP}.heartbeat"
GPU_CSV="$RUNTIME/gpu_monitor_${SLUG}_${STAMP}.csv"
PID_FILE="$RUNTIME/train_${SLUG}_${STAMP}.pid"
RUN_ENV="$RUNTIME/train_${SLUG}_${STAMP}.env"
READY="$RUNTIME/train_${SLUG}_${STAMP}.lock_ready"
SESSION_LOG="$SESSION_DIR/$SESSION.log"
LAST_CKPT="$SESSION_DIR/models/last_model-BetterLViT.pth.tar"
BEST_CKPT="$SESSION_DIR/models/best_model-BetterLViT.pth.tar"

export REPO PY LOCK STDOUT STDERR HEARTBEAT GPU_CSV PID_FILE RUN_ENV READY
export SESSION SESSION_LOG LAST_CKPT BEST_CKPT BRANCH EXPECTED_COMMIT

{
  printf 'STARTED_AT=%s\n' "$(date -Is)"
  printf 'BRANCH=%s\nCOMMIT=%s\nSESSION=%s\n' \
    "$BRANCH" "$EXPECTED_COMMIT" "$SESSION"
  printf 'SESSION_LOG=%s\nSTDOUT=%s\nSTDERR=%s\n' \
    "$SESSION_LOG" "$STDOUT" "$STDERR"
  printf 'HEARTBEAT=%s\nGPU_CSV=%s\n' "$HEARTBEAT" "$GPU_CSV"
  printf 'LAST_CHECKPOINT=%s\nBEST_CHECKPOINT=%s\n' \
    "$LAST_CKPT" "$BEST_CKPT"
  printf 'CHECKPOINT_SOURCE=FROM_SCRATCH\n'
  printf 'DATA_MANIFEST=%s\n' "$MANIFEST"
  printf 'DATA_MANIFEST_SHA256=%s\n' \
    "$(sha256sum "$MANIFEST" | awk '{print $1}')"
  printf 'SPLIT_PROTOCOL=%s\n' "$BETTERLVIT_SPLIT_PROTOCOL"
  printf 'SEED=1219\nBATCH_SIZE=16\nMAX_EPOCHS=200\n'
  printf 'LOSS_NAME=dice_focal\nDICE_FOCAL_WEIGHTS=0.5/0.5\n'
  printf 'FOCAL_GAMMA=2.0\nBOUNDARY_LOSS_WEIGHT=0.0\n'
  printf 'TEXT_MODALITY_DROPOUT_PROB=0.0\n'
  printf 'PYTHONHASHSEED=%s\nCUBLAS_WORKSPACE_CONFIG=%s\n' \
    "$PYTHONHASHSEED" "$CUBLAS_WORKSPACE_CONFIG"
} > "$RUN_ENV"

screen -DmS "$SCREEN" bash -c '
set -Eeuo pipefail
exec 9>"$LOCK"
flock 9
cd "$REPO"
printf "locked_at=%s\n" "$(date -Is)" > "$READY"

printf "timestamp,utilization_gpu_percent,memory_used_mib,memory_total_mib,power_draw_w,temperature_c\n" > "$GPU_CSV"

"$PY" -u train_model.py >"$STDOUT" 2>"$STDERR" &
train_pid=$!
printf "%s\n" "$train_pid" > "$PID_FILE"
printf "MAIN_PID=%s\n" "$train_pid" >> "$RUN_ENV"

(
  while kill -0 "$train_pid" 2>/dev/null; do
    now="$(date -Is)"
    gpu="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits | head -1)"
    printf "%s,%s\n" "$now" "$gpu" >> "$GPU_CSV"
    {
      printf "timestamp=%s\nstatus=RUNNING\nmain_pid=%s\n" "$now" "$train_pid"
      printf "session_log_bytes=%s\n" "$(stat -c %s "$SESSION_LOG" 2>/dev/null || printf 0)"
      printf "last_checkpoint_bytes=%s\n" "$(stat -c %s "$LAST_CKPT" 2>/dev/null || printf 0)"
      printf "gpu=%s\n" "$gpu"
    } > "${HEARTBEAT}.tmp"
    mv "${HEARTBEAT}.tmp" "$HEARTBEAT"
    sleep 10
  done
) &
monitor_pid=$!

set +e
wait "$train_pid"
rc=$?
set -e
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true

printf "timestamp=%s\nstatus=EXITED\nmain_pid=%s\nexit_code=%s\n" \
  "$(date -Is)" "$train_pid" "$rc" > "$HEARTBEAT"
exit "$rc"
' 8>&- 9>&-

# Release only the training lock. LAUNCH_LOCK remains held until this launcher
# verifies that the supervisor acquired the training lock and wrote READY.
flock -u 9
for _ in $(seq 1 50); do
  if test -s "$READY" && test -s "$PID_FILE"; then
    break
  fi
  sleep 0.2
done
test -s "$READY"
test -s "$PID_FILE"
screen -S "$SCREEN" -Q select . >/dev/null
TRAIN_PID="$(cat "$PID_FILE")"
[[ "$TRAIN_PID" =~ ^[0-9]+$ ]]
kill -0 "$TRAIN_PID"
for _ in $(seq 1 50); do
  if grep -qx 'status=RUNNING' "$HEARTBEAT" 2>/dev/null; then
    break
  fi
  kill -0 "$TRAIN_PID"
  sleep 0.2
done
grep -qx 'status=RUNNING' "$HEARTBEAT"
kill -0 "$TRAIN_PID"
HANDOFF_COMPLETE=1
printf 'LAUNCHED screen=%s pid=%s session=%s commit=%s\n' \
  "$SCREEN" "$TRAIN_PID" "$SESSION" "$EXPECTED_COMMIT"

#!/usr/bin/env bash
set -euo pipefail

a4_repo="/root/autodl-tmp/BetterLViT-paper-ablation"
a5_repo="/root/autodl-tmp/BetterLViT-paper-a5"
a4_expected="a1d40d3a305a34abc0e96885fae68532007485b2"
a5_tag="paper-a5-b16-seed1219-20260830"
a5_experiment="a5_lora_fmiseg_focal"
python_bin="/root/autodl-tmp/envs/betterlvit-paper/bin/python"
runtime_dir="$a5_repo/runtime_logs"
status_file="$runtime_dir/a4_to_a5_chain.status"
lock_file="$runtime_dir/a4_to_a5_chain.lock"

mkdir -p "$runtime_dir"

write_status() {
  local value="$1"
  printf '%s\n' "$value" >"${status_file}.tmp"
  mv -f "${status_file}.tmp" "$status_file"
}

read_env_value() {
  local file="$1"
  local key="$2"
  awk -F= -v key="$key" '$1 == key {sub(/^[^=]*=/, ""); print; exit}' "$file"
}

fail_chain() {
  local reason="$1"
  write_status "blocked_${reason}"
  echo "A4 to A5 chain blocked: $reason" >&2
  exit 1
}

exec 9>"$lock_file"
if ! flock -n 9; then
  echo "An A4 to A5 watcher is already active." >&2
  exit 4
fi

a5_expected="$(git -C "$a5_repo" rev-parse HEAD)"
[[ "$a5_expected" =~ ^[0-9a-f]{40}$ ]] || fail_chain "invalid_a5_commit"
test "$(git -C "$a5_repo" rev-list -n 1 "$a5_tag")" = "$a5_expected" \
  || fail_chain "a5_tag_mismatch"

write_status "waiting_for_a4"
while true; do
  a4_env="$a4_repo/runtime_logs/paper_experiment_current.env"
  [ -f "$a4_env" ] || fail_chain "a4_metadata_missing"
  test "$(read_env_value "$a4_env" GIT_COMMIT)" = "$a4_expected" \
    || fail_chain "a4_commit_mismatch"
  test "$(read_env_value "$a4_env" EXPERIMENT)" = "a4_lora_freq_focal" \
    || fail_chain "a4_experiment_mismatch"

  train_status_path="$(read_env_value "$a4_env" TRAIN_STATUS)"
  eval_status_path="$(read_env_value "$a4_env" EVAL_STATUS)"
  launcher_pid="$(read_env_value "$a4_env" LAUNCHER_PID)"
  train_state="$(cat "$train_status_path" 2>/dev/null || echo missing)"
  eval_state="$(cat "$eval_status_path" 2>/dev/null || echo missing)"

  if [ "$train_state" = "0" ] && [ "$eval_state" = "0" ]; then
    break
  fi
  case "$train_state" in
    running) ;;
    0) ;;
    *) fail_chain "a4_train_${train_state}" ;;
  esac
  case "$eval_state" in
    pending|running) ;;
    *) fail_chain "a4_eval_${eval_state}" ;;
  esac
  if ! kill -0 "$launcher_pid" 2>/dev/null; then
    fail_chain "a4_launcher_missing"
  fi
  sleep 60
done

write_status "a4_complete_preflight"
if pgrep -af '[t]rain_model.py|[e]valuate_experiment.py' >/dev/null; then
  fail_chain "gpu_process_still_active"
fi
test -z "$(git -C "$a5_repo" status --porcelain --untracked-files=no)" \
  || fail_chain "a5_worktree_dirty"
test "$(git -C "$a5_repo" rev-parse HEAD)" = "$a5_expected" \
  || fail_chain "a5_head_changed"

export HF_HOME="/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_MODULES_CACHE="$HF_HOME/modules"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export BETTERLVIT_DETERMINISTIC=1
export BETTERLVIT_CUDNN_ENABLED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=1219

for run in 1 2; do
  write_status "a5_preflight_${run}"
  "$python_bin" "$a5_repo/tools/smoke_paper_profile.py" \
    --experiment "$a5_experiment" \
    --batch-size 16 \
    --seed 1219 \
    >"$runtime_dir/a5_preflight_${run}.stdout.log" \
    2>"$runtime_dir/a5_preflight_${run}.stderr.log" \
    || fail_chain "a5_preflight_${run}_failed"
done

write_status "launching_a5"
cd "$a5_repo"
BETTERLVIT_EVAL_BATCH_SIZE=16 \
  bash scripts/start_paper_experiment_server.sh \
  "$a5_experiment" 1219 150 16 \
  >"$runtime_dir/a5_chain_launch.stdout.log" \
  2>"$runtime_dir/a5_chain_launch.stderr.log" \
  || fail_chain "a5_launch_failed"

a5_env="$runtime_dir/paper_experiment_current.env"
test "$(read_env_value "$a5_env" GIT_COMMIT)" = "$a5_expected" \
  || fail_chain "a5_runtime_commit_mismatch"
test "$(read_env_value "$a5_env" EXPERIMENT)" = "$a5_experiment" \
  || fail_chain "a5_runtime_experiment_mismatch"
a5_launcher_pid="$(read_env_value "$a5_env" LAUNCHER_PID)"
kill -0 "$a5_launcher_pid" 2>/dev/null || fail_chain "a5_launcher_not_alive"
write_status "a5_started"
echo "A5 started successfully from commit $a5_expected"

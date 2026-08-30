#!/usr/bin/env bash
set -uo pipefail

if [ "$#" -ne 5 ]; then
  echo "Usage: $0 A6_REPO A7_REPO EXPECTED_A6_COMMIT EXPECTED_A7_COMMIT CHAIN_STATUS" >&2
  exit 2
fi

a6_repo="$1"
a7_repo="$2"
expected_a6_commit="$3"
expected_a7_commit="$4"
chain_status="$5"
python_bin="${BETTERLVIT_PYTHON:-/root/autodl-tmp/envs/betterlvit-paper/bin/python}"
hf_home="${BETTERLVIT_HF_HOME:-/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface}"
runtime_dir="$a7_repo/runtime_logs"
chain_log="$runtime_dir/a6_to_a7_chain.log"

write_status() {
  local value="$1"
  local temporary="${chain_status}.tmp"
  printf '%s\n' "$value" >"$temporary"
  mv -f "$temporary" "$chain_status"
  printf '%s %s\n' "$(date -Is)" "$value" >>"$chain_log"
}

fail_chain() {
  local value="$1"
  write_status "$value"
  exit 1
}

mkdir -p "$runtime_dir"
: >"$chain_log"
write_status waiting_a6

a6_env="$a6_repo/runtime_logs/paper_experiment_current.env"
while [ ! -s "$a6_env" ]; do sleep 30; done

# shellcheck disable=SC1090
. "$a6_env"
[ "${EXPERIMENT:-}" = "a6_tcsr" ] || fail_chain blocked_a6_experiment_mismatch
[ "${GIT_COMMIT:-}" = "$expected_a6_commit" ] || fail_chain blocked_a6_commit_mismatch
[ "${EPOCHS:-}" = "150" ] || fail_chain blocked_a6_epochs_mismatch
[ "${BATCH_SIZE:-}" = "16" ] || fail_chain blocked_a6_batch_mismatch
[ "${SEED:-}" = "1219" ] || fail_chain blocked_a6_seed_mismatch

while :; do
  train_value="$(tr -d '\r\n ' <"$TRAIN_STATUS" 2>/dev/null || true)"
  eval_value="$(tr -d '\r\n ' <"$EVAL_STATUS" 2>/dev/null || true)"
  if [ "$train_value" = "0" ] && [ "$eval_value" = "0" ]; then
    break
  fi
  case "$train_value" in
    running|'') ;;
    0) ;;
    *) fail_chain "blocked_a6_training_${train_value}" ;;
  esac
  case "$eval_value" in
    pending|running|'') ;;
    0) ;;
    *) fail_chain "blocked_a6_evaluation_${eval_value}" ;;
  esac
  sleep 30
done
write_status a6_train_eval_succeeded

a6_result="$a6_repo/Covid19/BetterLViT/a6_tcsr/A6_Test_session_08.30_18h48/a6_tcsr_evaluation.json"
[ -s "$a6_result" ] || fail_chain blocked_a6_result_missing
"$python_bin" -c '
import json, sys
p, expected = sys.argv[1:3]
x = json.load(open(p, "r", encoding="utf-8"))
assert x.get("git_commit") == expected, x.get("git_commit")
assert x.get("experiment_name") == "a6_tcsr", x.get("experiment_name")
assert x.get("text_use_lora") is False, x.get("text_use_lora")
assert x.get("test", {}).get("samples") == 2113, x.get("test", {}).get("samples")
for key in ("threshold_0_5", "validation_selected_threshold"):
    metrics = x["test"][key]
    for metric in ("dice", "iou", "micro_dice", "micro_iou"):
        assert isinstance(metrics.get(metric), (int, float)), (key, metric)
' "$a6_result" "$expected_a6_commit" || fail_chain blocked_a6_result_invalid
write_status a6_result_verified

if pgrep -af '[t]rain_model.py|[e]valuate_experiment.py' >/dev/null; then
  fail_chain blocked_gpu_process_still_active
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
  fail_chain blocked_gpu_not_idle
fi

[ "$(git -C "$a7_repo" rev-parse HEAD)" = "$expected_a7_commit" ] || fail_chain blocked_a7_commit_mismatch
[ -z "$(git -C "$a7_repo" status --porcelain)" ] || fail_chain blocked_a7_worktree_dirty

export HF_HOME="$hf_home"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_MODULES_CACHE="$HF_HOME/modules"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for attempt in 1 2; do
  write_status "a7_preflight_${attempt}_running"
  preflight_out="$runtime_dir/a7_exact_b16_preflight_${attempt}.json"
  preflight_err="$runtime_dir/a7_exact_b16_preflight_${attempt}.stderr.log"
  (
    cd "$a7_repo" &&
    "$python_bin" tools/smoke_paper_profile.py \
      --experiment a7_tcsr_freq \
      --batch-size 16 \
      --seed 1219
  ) >"$preflight_out" 2>"$preflight_err"
  rc=$?
  printf '%s\n' "$rc" >"$runtime_dir/a7_exact_b16_preflight_${attempt}.status"
  [ "$rc" -eq 0 ] || fail_chain "blocked_a7_preflight_${attempt}_rc_${rc}"
  "$python_bin" -c '
import json, sys
x = json.load(open(sys.argv[1], "r", encoding="utf-8"))
assert x.get("experiment") == "a7_tcsr_freq", x.get("experiment")
assert x.get("batch_size") == 16, x.get("batch_size")
assert x.get("text_use_lora") is False, x.get("text_use_lora")
assert x.get("trainable_text_tensors") == 0, x.get("trainable_text_tensors")
assert x.get("lora_parameter_tensors") == 0, x.get("lora_parameter_tensors")
assert x.get("status") == "ok", x.get("status")
' "$preflight_out" || fail_chain "blocked_a7_preflight_${attempt}_invalid"
  write_status "a7_preflight_${attempt}_succeeded"
done

if pgrep -af '[t]rain_model.py|[e]valuate_experiment.py' >/dev/null; then
  fail_chain blocked_process_before_a7_launch
fi

write_status a7_launching
(
  cd "$a7_repo" &&
  BETTERLVIT_EVAL_BATCH_SIZE=16 \
    ./scripts/start_paper_experiment_server.sh a7_tcsr_freq 1219 150 16
) >>"$chain_log" 2>&1 || fail_chain blocked_a7_launch_failed

a7_env="$a7_repo/runtime_logs/paper_experiment_current.env"
[ -s "$a7_env" ] || fail_chain blocked_a7_runtime_metadata_missing
# shellcheck disable=SC1090
. "$a7_env"
[ "${EXPERIMENT:-}" = "a7_tcsr_freq" ] || fail_chain blocked_a7_runtime_experiment_mismatch
[ "${GIT_COMMIT:-}" = "$expected_a7_commit" ] || fail_chain blocked_a7_runtime_commit_mismatch
[ "$(tr -d '\r\n ' <"$TRAIN_STATUS" 2>/dev/null || true)" = "running" ] || fail_chain blocked_a7_not_running
[ "$(tr -d '\r\n ' <"$EVAL_STATUS" 2>/dev/null || true)" = "pending" ] || fail_chain blocked_a7_eval_not_pending
write_status a7_started

#!/usr/bin/env bash
set -uo pipefail

if [ "$#" -ne 10 ]; then
  echo "Usage: $0 C2_REPO P7_REPO C2_SHA P7_SHA PYTHON SEED EPOCHS BATCH STATUS OUTPUT_DIR" >&2
  exit 2
fi
c2_repo="$1"
p7_repo="$2"
c2_sha="$3"
p7_sha="$4"
python_bin="$5"
seed="$6"
epochs="$7"
batch_size="$8"
status_file="$9"
output_dir="${10}"

write_status() {
  local value="$1"
  printf '%s\n' "$value" >"${status_file}.tmp"
  mv -f "${status_file}.tmp" "$status_file"
}
fail() {
  write_status "$1"
  echo "$1" >&2
  exit 1
}
latest_best() {
  local repo="$1"
  local experiment="$2"
  find "$repo/Covid19/BetterLViT/$experiment" -type f \
    -name 'best_model-BetterLViT.pth.tar' -printf '%T@|%p\n' \
    | sort -n | tail -1 | cut -d'|' -f2-
}
configure_env() {
  local experiment="$1"
  local sha="$2"
  export BETTERLVIT_EXPERIMENT="$experiment"
  export BETTERLVIT_SEED="$seed"
  export BETTERLVIT_EPOCHS="$epochs"
  export BETTERLVIT_BATCH_SIZE="$batch_size"
  export BETTERLVIT_TRAIN_DROP_LAST=1
  export BETTERLVIT_NUM_WORKERS=4
  export BETTERLVIT_DETERMINISTIC=1
  export BETTERLVIT_CUDNN_ENABLED=1
  export BETTERLVIT_VIS_FREQUENCY=100000
  export BETTERLVIT_GIT_COMMIT="$sha"
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  export PYTHONHASHSEED="$seed"
  export HF_HOME="${BETTERLVIT_HF_HOME:-/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface}"
  export HF_HUB_CACHE="$HF_HOME/hub"
  export HF_MODULES_CACHE="$HF_HOME/modules"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false
}

mkdir -p "$output_dir" "$(dirname "$status_file")"
write_status starting
exec 9>/root/autodl-tmp/betterlvit_cdrr_pair.lock
flock -n 9 || fail lock_unavailable
[ "$(git -C "$c2_repo" rev-parse HEAD)" = "$c2_sha" ] || fail c2_commit_mismatch
[ "$(git -C "$p7_repo" rev-parse HEAD)" = "$p7_sha" ] || fail p7_commit_mismatch
[ -z "$(git -C "$c2_repo" status --porcelain --untracked-files=no)" ] || fail c2_dirty_worktree
[ -z "$(git -C "$p7_repo" status --porcelain --untracked-files=no)" ] || fail p7_dirty_worktree

write_status c2_training
configure_env c2_cdrr_control "$c2_sha"
(cd "$c2_repo" && "$python_bin" train_model.py) \
  >"$output_dir/c2_train.stdout.log" 2>"$output_dir/c2_train.stderr.log"
c2_rc=$?
[ "$c2_rc" -eq 0 ] || fail "c2_training_failed_rc_${c2_rc}"
c2_checkpoint="$(latest_best "$c2_repo" c2_cdrr_control)"
[ -n "$c2_checkpoint" ] && [ -f "$c2_checkpoint" ] || fail c2_checkpoint_missing

write_status c2_exporting
(cd "$c2_repo" && "$python_bin" tools/export_validation_metrics.py \
  --experiment c2_cdrr_control --checkpoint "$c2_checkpoint" \
  --output "$output_dir/c2_validation.json" --batch-size "$batch_size" \
  --threshold 0.5) >"$output_dir/c2_export.stdout.log" \
  2>"$output_dir/c2_export.stderr.log"
c2_export_rc=$?
[ "$c2_export_rc" -eq 0 ] || fail "c2_export_failed_rc_${c2_export_rc}"

write_status p7_training
configure_env p7_cdrr_v1 "$p7_sha"
(cd "$p7_repo" && "$python_bin" train_model.py) \
  >"$output_dir/p7_train.stdout.log" 2>"$output_dir/p7_train.stderr.log"
p7_rc=$?
[ "$p7_rc" -eq 0 ] || fail "p7_training_failed_rc_${p7_rc}"
p7_checkpoint="$(latest_best "$p7_repo" p7_cdrr_v1)"
[ -n "$p7_checkpoint" ] && [ -f "$p7_checkpoint" ] || fail p7_checkpoint_missing

write_status p7_exporting
(cd "$p7_repo" && "$python_bin" tools/export_validation_metrics.py \
  --experiment p7_cdrr_v1 --checkpoint "$p7_checkpoint" \
  --output "$output_dir/p7_validation.json" --batch-size "$batch_size" \
  --threshold 0.5) >"$output_dir/p7_export.stdout.log" \
  2>"$output_dir/p7_export.stderr.log"
p7_export_rc=$?
[ "$p7_export_rc" -eq 0 ] || fail "p7_export_failed_rc_${p7_export_rc}"

write_status comparing
(cd "$p7_repo" && "$python_bin" tools/compare_validation_metrics.py \
  --control "$output_dir/c2_validation.json" \
  --candidate "$output_dir/p7_validation.json" \
  --output "$output_dir/c2_vs_p7.json" --seed "$seed") \
  >"$output_dir/compare.stdout.log" 2>"$output_dir/compare.stderr.log"
compare_rc=$?
[ "$compare_rc" -eq 0 ] || fail "comparison_failed_rc_${compare_rc}"
write_status complete

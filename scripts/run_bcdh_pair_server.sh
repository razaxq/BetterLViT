#!/usr/bin/env bash
set -uo pipefail

if [ "$#" -ne 10 ]; then
  echo "Usage: $0 C1_REPO P6_REPO C1_SHA P6_SHA PYTHON SEED EPOCHS BATCH STATUS OUTPUT_DIR" >&2
  exit 2
fi
c1_repo="$1"
p6_repo="$2"
c1_sha="$3"
p6_sha="$4"
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
exec 9>/root/autodl-tmp/betterlvit_bcdh_pair.lock
flock -n 9 || fail lock_unavailable
[ "$(git -C "$c1_repo" rev-parse HEAD)" = "$c1_sha" ] || fail c1_commit_mismatch
[ "$(git -C "$p6_repo" rev-parse HEAD)" = "$p6_sha" ] || fail p6_commit_mismatch
[ -z "$(git -C "$c1_repo" status --porcelain --untracked-files=no)" ] || fail c1_dirty_worktree
[ -z "$(git -C "$p6_repo" status --porcelain --untracked-files=no)" ] || fail p6_dirty_worktree

write_status c1_training
configure_env c1_bcdh_control "$c1_sha"
(cd "$c1_repo" && "$python_bin" train_model.py) \
  >"$output_dir/c1_train.stdout.log" 2>"$output_dir/c1_train.stderr.log"
c1_rc=$?
[ "$c1_rc" -eq 0 ] || fail "c1_training_failed_rc_${c1_rc}"
c1_checkpoint="$(latest_best "$c1_repo" c1_bcdh_control)"
[ -n "$c1_checkpoint" ] && [ -f "$c1_checkpoint" ] || fail c1_checkpoint_missing

write_status c1_exporting
(cd "$c1_repo" && "$python_bin" tools/export_validation_metrics.py \
  --experiment c1_bcdh_control --checkpoint "$c1_checkpoint" \
  --output "$output_dir/c1_validation.json" --batch-size "$batch_size" \
  --threshold 0.5) >"$output_dir/c1_export.stdout.log" \
  2>"$output_dir/c1_export.stderr.log"
c1_export_rc=$?
[ "$c1_export_rc" -eq 0 ] || fail "c1_export_failed_rc_${c1_export_rc}"

write_status p6_training
configure_env p6_bcdh_r_v1 "$p6_sha"
(cd "$p6_repo" && "$python_bin" train_model.py) \
  >"$output_dir/p6_train.stdout.log" 2>"$output_dir/p6_train.stderr.log"
p6_rc=$?
[ "$p6_rc" -eq 0 ] || fail "p6_training_failed_rc_${p6_rc}"
p6_checkpoint="$(latest_best "$p6_repo" p6_bcdh_r_v1)"
[ -n "$p6_checkpoint" ] && [ -f "$p6_checkpoint" ] || fail p6_checkpoint_missing

write_status p6_exporting
(cd "$p6_repo" && "$python_bin" tools/export_validation_metrics.py \
  --experiment p6_bcdh_r_v1 --checkpoint "$p6_checkpoint" \
  --output "$output_dir/p6_validation.json" --batch-size "$batch_size" \
  --threshold 0.5) >"$output_dir/p6_export.stdout.log" \
  2>"$output_dir/p6_export.stderr.log"
p6_export_rc=$?
[ "$p6_export_rc" -eq 0 ] || fail "p6_export_failed_rc_${p6_export_rc}"

write_status comparing
(cd "$p6_repo" && "$python_bin" tools/compare_validation_metrics.py \
  --control "$output_dir/c1_validation.json" \
  --candidate "$output_dir/p6_validation.json" \
  --output "$output_dir/c1_vs_p6.json" --seed "$seed") \
  >"$output_dir/compare.stdout.log" 2>"$output_dir/compare.stderr.log"
compare_rc=$?
[ "$compare_rc" -eq 0 ] || fail "comparison_failed_rc_${compare_rc}"
write_status complete

#!/usr/bin/env bash
set -uo pipefail

if [ "$#" -ne 10 ]; then
  echo "Usage: $0 C3_REPO P8_REPO C3_SHA P8_SHA PYTHON SEED EPOCHS BATCH STATUS OUTPUT_DIR" >&2
  exit 2
fi
c3_repo="$1"
p8_repo="$2"
c3_sha="$3"
p8_sha="$4"
python_bin="$5"
seed="$6"
epochs="$7"
batch_size="$8"
status_file="$9"
output_dir="${10}"

write_status() {
  printf '%s\n' "$1" >"${status_file}.tmp"
  mv -f "${status_file}.tmp" "$status_file"
}
fail() {
  write_status "$1"
  echo "$1" >&2
  exit 1
}
latest_best() {
  find "$1/Covid19/BetterLViT/$2" -type f \
    -name 'best_model-BetterLViT.pth.tar' -printf '%T@|%p\n' \
    | sort -n | tail -1 | cut -d'|' -f2-
}
configure_env() {
  export BETTERLVIT_EXPERIMENT="$1"
  export BETTERLVIT_GIT_COMMIT="$2"
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
  export HF_HOME="${BETTERLVIT_HF_HOME:-/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface}"
  export HF_HUB_CACHE="$HF_HOME/hub"
  export HF_MODULES_CACHE="$HF_HOME/modules"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false
}

mkdir -p "$output_dir" "$(dirname "$status_file")"
write_status starting
exec 9>/root/autodl-tmp/betterlvit_race_pair.lock
flock -n 9 || fail lock_unavailable
[ "$(git -C "$c3_repo" rev-parse HEAD)" = "$c3_sha" ] || fail c3_commit_mismatch
[ "$(git -C "$p8_repo" rev-parse HEAD)" = "$p8_sha" ] || fail p8_commit_mismatch
[ -z "$(git -C "$c3_repo" status --porcelain --untracked-files=no)" ] || fail c3_dirty_worktree
[ -z "$(git -C "$p8_repo" status --porcelain --untracked-files=no)" ] || fail p8_dirty_worktree

write_status c3_training
configure_env c3_race_control "$c3_sha"
(cd "$c3_repo" && "$python_bin" train_model.py) \
  >"$output_dir/c3_train.stdout.log" 2>"$output_dir/c3_train.stderr.log"
rc=$?
[ "$rc" -eq 0 ] || fail "c3_training_failed_rc_${rc}"
c3_checkpoint="$(latest_best "$c3_repo" c3_race_control)"
[ -n "$c3_checkpoint" ] && [ -f "$c3_checkpoint" ] || fail c3_checkpoint_missing

write_status c3_exporting
(cd "$c3_repo" && "$python_bin" tools/export_validation_metrics.py \
  --experiment c3_race_control --checkpoint "$c3_checkpoint" \
  --output "$output_dir/c3_validation.json" --batch-size "$batch_size" \
  --threshold 0.5) >"$output_dir/c3_export.stdout.log" \
  2>"$output_dir/c3_export.stderr.log"
rc=$?
[ "$rc" -eq 0 ] || fail "c3_export_failed_rc_${rc}"

write_status p8_training
configure_env p8_race_fuse_v1 "$p8_sha"
(cd "$p8_repo" && "$python_bin" train_model.py) \
  >"$output_dir/p8_train.stdout.log" 2>"$output_dir/p8_train.stderr.log"
rc=$?
[ "$rc" -eq 0 ] || fail "p8_training_failed_rc_${rc}"
p8_checkpoint="$(latest_best "$p8_repo" p8_race_fuse_v1)"
[ -n "$p8_checkpoint" ] && [ -f "$p8_checkpoint" ] || fail p8_checkpoint_missing

write_status p8_exporting
(cd "$p8_repo" && "$python_bin" tools/export_validation_metrics.py \
  --experiment p8_race_fuse_v1 --checkpoint "$p8_checkpoint" \
  --output "$output_dir/p8_validation.json" --batch-size "$batch_size" \
  --threshold 0.5) >"$output_dir/p8_export.stdout.log" \
  2>"$output_dir/p8_export.stderr.log"
rc=$?
[ "$rc" -eq 0 ] || fail "p8_export_failed_rc_${rc}"

write_status comparing
(cd "$p8_repo" && "$python_bin" tools/compare_validation_metrics.py \
  --control "$output_dir/c3_validation.json" \
  --candidate "$output_dir/p8_validation.json" \
  --output "$output_dir/c3_vs_p8.json" --seed "$seed") \
  >"$output_dir/compare.stdout.log" 2>"$output_dir/compare.stderr.log"
rc=$?
[ "$rc" -eq 0 ] || fail "comparison_failed_rc_${rc}"
write_status complete

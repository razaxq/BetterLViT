#!/usr/bin/env bash
set -uo pipefail

if [ "$#" -ne 9 ]; then
  echo "Usage: $0 REPO LOCK PYTHON EXPERIMENT EVAL_BATCH TRAIN_STATUS EVAL_STATUS EVAL_STDOUT EVAL_STDERR" >&2
  exit 2
fi

repo_root="$1"
lock_file="$2"
python_bin="$3"
experiment="$4"
evaluation_batch_size="$5"
training_status="$6"
evaluation_status="$7"
evaluation_stdout="$8"
evaluation_stderr="$9"

write_status() {
  status_path="$1"
  status_value="$2"
  temporary_path="${status_path}.tmp"
  printf '%s\n' "$status_value" >"$temporary_path"
  mv -f "$temporary_path" "$status_path"
}

mkdir -p "$(dirname "$training_status")"
write_status "$training_status" running
write_status "$evaluation_status" pending

exec 9>"$lock_file"
if ! flock -n 9; then
  write_status "$training_status" lock_unavailable
  write_status "$evaluation_status" skipped_lock_unavailable
  echo "Training/evaluation lock is already held: $lock_file" >&2
  exit 4
fi

cd "$repo_root" || {
  write_status "$training_status" repo_unavailable
  write_status "$evaluation_status" skipped_repo_unavailable
  exit 5
}

"$python_bin" train_model.py
training_rc=$?
write_status "$training_status" "$training_rc"
if [ "$training_rc" -ne 0 ]; then
  write_status "$evaluation_status" "skipped_training_rc_${training_rc}"
  exit "$training_rc"
fi

write_status "$evaluation_status" running
"$python_bin" tools/evaluate_experiment.py \
  --experiment "$experiment" \
  --batch-size "$evaluation_batch_size" \
  >"$evaluation_stdout" 2>"$evaluation_stderr"
evaluation_rc=$?
write_status "$evaluation_status" "$evaluation_rc"
exit "$evaluation_rc"

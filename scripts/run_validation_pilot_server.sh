#!/usr/bin/env bash
set -uo pipefail

if [ "$#" -ne 5 ]; then
  echo "Usage: $0 REPO LOCK PYTHON STATUS EXPERIMENT" >&2
  exit 2
fi

repo_root="$1"
lock_file="$2"
python_bin="$3"
status_file="$4"
experiment="$5"

write_status() {
  local value="$1"
  local temporary_path="${status_file}.tmp"
  printf '%s\n' "$value" >"$temporary_path"
  mv -f "$temporary_path" "$status_file"
}

mkdir -p "$(dirname "$status_file")"
write_status running

exec 9>"$lock_file"
if ! flock -n 9; then
  write_status lock_unavailable
  echo "Training lock is already held: $lock_file" >&2
  exit 4
fi

cd "$repo_root" || {
  write_status repo_unavailable
  exit 5
}

if [ "$experiment" != "p1_tcsrv21_boundary_router" ] \
  && [ "$experiment" != "p2_tcsrv22_single_hop_boundary" ] \
  && [ "$experiment" != "p3_tcsrv23_calibrated_gate" ]; then
  if [ "$experiment" = "p4_tcsrv24_sparse_boundary" ]; then
    :
  else
  write_status unsupported_experiment
  echo "Validation-only runner refuses experiment: $experiment" >&2
  exit 6
  fi
fi

"$python_bin" train_model.py
training_rc=$?
write_status "$training_rc"
exit "$training_rc"

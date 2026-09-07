#!/usr/bin/env bash
set -euo pipefail

metadata=/root/autodl-tmp/BetterLViT-paper-p7-cdrr/runtime_logs/cdrr_pair_current.env
cat "$metadata"
status_file=$(sed -n 's/^STATUS=//p' "$metadata")
output_dir=$(sed -n 's/^OUTPUT_DIR=//p' "$metadata")
printf 'CHAIN_STATUS=%s\n' "$(cat "$status_file")"
pgrep -af '[t]rain_model.py|[r]un_cdrr_pair_server.sh' || true
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader
tail -40 "$output_dir/c2_train.stdout.log" || true
tail -40 "$output_dir/c2_train.stderr.log" || true

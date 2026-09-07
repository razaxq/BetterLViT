#!/usr/bin/env bash
set -euo pipefail

shared_root=/root/autodl-fs/betterlvit_5090_migration
shared_physical=/autodl-fs/data/betterlvit_5090_migration
legacy_root="$shared_root/BetterLViT/Covid19/BetterLViT"
manifest="$shared_root/runtime_logs/storage_optimization_20260903_deleted_legacy_last.sha256"

last_files=(
  "$legacy_root/Test_session_08.12_20h04/models/last_model-BetterLViT.pth.tar"
  "$legacy_root/Test_session_08.13_12h01/models/last_model-BetterLViT.pth.tar"
  "$legacy_root/Test_session_08.13_12h37/models/last_model-BetterLViT.pth.tar"
  "$legacy_root/Test_session_08.13_20h05/models/last_model-BetterLViT.pth.tar"
  "$legacy_root/Test_session_08.14_11h46/models/last_model-BetterLViT.pth.tar"
  "$legacy_root/Test_session_08.14_20h03/models/last_model-BetterLViT.pth.tar"
)

mkdir -p "$(dirname "$manifest")"
{
  printf '# Storage optimization at %s\n' "$(date -Is)"
  printf '# Removed legacy Last checkpoints only when a sibling Best checkpoint existed.\n'
  printf '# format: sha256 bytes path\n'
} >"$manifest"

for last in "${last_files[@]}"; do
  resolved=$(realpath "$last")
  case "$resolved" in
    "$shared_physical"/BetterLViT/Covid19/BetterLViT/Test_session_*/models/last_model-BetterLViT.pth.tar) ;;
    *) echo "Unsafe legacy checkpoint path: $resolved" >&2; exit 2 ;;
  esac
  best=${last%/last_model-BetterLViT.pth.tar}/best_model-BetterLViT.pth.tar
  test -f "$last"
  test -f "$best"
  hash=$(sha256sum "$last" | awk '{print $1}')
  bytes=$(stat -c %s "$last")
  printf '%s %s %s\n' "$hash" "$bytes" "$last" >>"$manifest"
done

for last in "${last_files[@]}"; do
  rm -- "$last"
  test ! -e "$last"
done

remove_tree() {
  local expected=$1
  local allowed=$2
  local resolved
  test -d "$expected"
  resolved=$(realpath "$expected")
  test "$resolved" = "$allowed"
  rm -rf -- "$resolved"
  test ! -e "$resolved"
  printf 'REMOVED_TREE=%s\n' "$resolved"
}

remove_tree "$shared_root/huggingface-cache" "$shared_physical/huggingface-cache"
remove_tree /root/autodl-tmp/hf-upload-staging /root/autodl-tmp/hf-upload-staging
remove_tree /root/autodl-tmp/wheelhouse-cu128 /root/autodl-tmp/wheelhouse-cu128
remove_tree /root/autodl-tmp/huggingface /root/autodl-tmp/huggingface

sync
printf 'MANIFEST=%s\n' "$manifest"
printf 'MANIFEST_LINES=%s\n' "$(grep -vc '^#' "$manifest")"
du -sb "$shared_root"
du -sh "$shared_root"
df -h /root/autodl-tmp /root/autodl-fs
status_file=/root/autodl-tmp/BetterLViT-paper-p7-cdrr/runtime_logs/cdrr_pair_20260903_122723/chain.status
printf 'CHAIN_STATUS=%s\n' "$(cat "$status_file")"
pgrep -af '[t]rain_model.py|[r]un_cdrr_pair_server.sh' || true
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader

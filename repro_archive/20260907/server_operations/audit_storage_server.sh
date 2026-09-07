#!/usr/bin/env bash
set -euo pipefail

pid=$(pgrep -n -f '[t]rain_model.py')
printf 'TRAIN_PID=%s\n' "$pid"
tr '\0' '\n' <"/proc/$pid/environ" | grep -E '^(HF_HOME|HF_HUB_CACHE|BETTERLVIT_EXPERIMENT|BETTERLVIT_GIT_COMMIT)='
find /root/autodl-fs/betterlvit_5090_migration -xdev -type f -name archive.zip -printf '%s %p\n'
find /root/autodl-fs/betterlvit_5090_migration/BetterLViT/Covid19/BetterLViT \
  -mindepth 3 -maxdepth 3 -type f -path '*/models/last_model-BetterLViT.pth.tar' \
  -printf '%s %p\n' | sort
printf 'OLD_LAST_WITH_BEST\n'
for last in /root/autodl-fs/betterlvit_5090_migration/BetterLViT/Covid19/BetterLViT/Test_session_*/models/last_model-BetterLViT.pth.tar; do
  best=${last%/last_model-BetterLViT.pth.tar}/best_model-BetterLViT.pth.tar
  if [ -f "$last" ] && [ -f "$best" ]; then
    printf '%s %s\n' "$(stat -c %s "$last")" "$last"
  fi
done
printf 'ACTIVE_OPEN_FILES_IN_CLEANUP_PATHS\n'
for candidate in \
  /root/autodl-fs/betterlvit_5090_migration/huggingface-cache \
  /root/autodl-tmp/hf-upload-staging \
  /root/autodl-tmp/wheelhouse-cu128 \
  /root/autodl-tmp/huggingface; do
  if command -v lsof >/dev/null 2>&1; then
    lsof +D "$candidate" 2>/dev/null | head -20 || true
  fi
done

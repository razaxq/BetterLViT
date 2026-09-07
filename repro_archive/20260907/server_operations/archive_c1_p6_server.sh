#!/usr/bin/env bash
set -euo pipefail

archive_root=/root/autodl-fs/betterlvit_5090_migration/paper_experiment_artifacts
c1_commit=106aeab700ee98653b5ec27994a0aa15b60dac07
p6_commit=7217660e6ef16e2a495bab4f20c73403468f55e1
c1_repo=/root/autodl-tmp/BetterLViT-paper-c1-bcdh
p6_repo=/root/autodl-tmp/BetterLViT-paper-p6-bcdh
c1_session="$c1_repo/Covid19/BetterLViT/c1_bcdh_control/C1_Test_session_09.02_20h41"
p6_session="$p6_repo/Covid19/BetterLViT/p6_bcdh_r_v1/P6_Test_session_09.02_23h07"
c1_dest="$archive_root/$c1_commit"
p6_dest="$archive_root/$p6_commit"

for path in "$c1_session" "$p6_session"; do
  resolved=$(realpath "$path")
  case "$resolved" in
    /root/autodl-tmp/BetterLViT-paper-c1-bcdh/Covid19/BetterLViT/c1_bcdh_control/C1_Test_session_09.02_20h41|\
    /root/autodl-tmp/BetterLViT-paper-p6-bcdh/Covid19/BetterLViT/p6_bcdh_r_v1/P6_Test_session_09.02_23h07) ;;
    *) echo "Unsafe session path: $resolved" >&2; exit 2 ;;
  esac
done

mkdir -p "$c1_dest/session" "$c1_dest/runtime_logs" "$c1_dest/manifest"
mkdir -p "$p6_dest/session" "$p6_dest/runtime_logs" "$p6_dest/manifest"
rsync -a --delete "$c1_session/" "$c1_dest/session/"
install -m 0644 "$c1_repo/experiment_manifests/c1_bcdh_control.json" "$c1_dest/manifest/"
install -m 0644 "$p6_repo/runtime_logs/bcdh_pair_20260902_204104/c1_validation.json" "$c1_dest/runtime_logs/"
rsync -a --delete "$p6_session/" "$p6_dest/session/"
rsync -a --delete "$p6_repo/runtime_logs/" "$p6_dest/runtime_logs/"
install -m 0644 "$p6_repo/experiment_manifests/p6_bcdh_r_v1.json" "$p6_dest/manifest/"

tree_stats() {
  local path=$1
  printf '%s files=' "$path"
  find "$path" -type f | wc -l
  printf '%s bytes=' "$path"
  find "$path" -type f -printf '%s\n' | awk '{sum += $1} END {print sum+0}'
}

tree_stats "$c1_session"
tree_stats "$c1_dest/session"
tree_stats "$p6_session"
tree_stats "$p6_dest/session"

c1_last_expected=0727995756246927f2f51ebf95a694377f1fec404bc74bf5f07668d0dd0a1755
c1_best_expected=cfe9a9274e768c01254baa77cc18dc3fe7c1a2e6c3d4c86a2b92670379623bcf
p6_last_expected=b8c33fee055adf6a7f2350ec0f1768d215efb7125f31a563b54d5023ac402654
p6_best_expected=3f60b8eba8a8f10fac755c06e8d81cfd66cc36fab907d94f6cc0e34ef270a4c7

check_hash() {
  local expected=$1
  local path=$2
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  test "$actual" = "$expected"
  printf '%s  %s\n' "$actual" "$path"
}

check_hash "$c1_last_expected" "$c1_dest/session/models/last_model-BetterLViT.pth.tar"
check_hash "$c1_best_expected" "$c1_dest/session/models/best_model-BetterLViT.pth.tar"
check_hash "$p6_last_expected" "$p6_dest/session/models/last_model-BetterLViT.pth.tar"
check_hash "$p6_best_expected" "$p6_dest/session/models/best_model-BetterLViT.pth.tar"

for relative in bcdh_pair_20260902_204104/c1_validation.json bcdh_pair_20260902_204104/p6_validation.json bcdh_pair_20260902_204104/c1_vs_p6.json bcdh_frequency_diagnostic/p6_frequency_heads.json; do
  test "$(sha256sum "$p6_repo/runtime_logs/$relative" | awk '{print $1}')" = "$(sha256sum "$p6_dest/runtime_logs/$relative" | awk '{print $1}')"
done

rm -rf -- "$c1_session"
rm -rf -- "$p6_session"

test ! -e "$c1_session"
test ! -e "$p6_session"
df -h /root/autodl-tmp /root/autodl-fs

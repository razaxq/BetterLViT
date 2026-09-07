#!/usr/bin/env bash
set -euo pipefail

bundle=/root/cdrr-pilots.bundle
c2_dir=/root/autodl-tmp/BetterLViT-paper-c2-cdrr
p7_dir=/root/autodl-tmp/BetterLViT-paper-p7-cdrr
c2_commit=06479cd3302a8ca11022eac0a6b62bdad097eb65
p7_commit=fe4547a0c60fe948c9a574d9afc7d691370aeb42
c2_tag=pilot-c2-cdrr-control-frozen-b16-seed1219-20260903
p7_tag=pilot-p7-cdrr-v1-frozen-b16-seed1219-20260903
dataset=/root/autodl-tmp/datasets/Covid19

test -f "$bundle"
test -d "$dataset"
test ! -e "$c2_dir"
test ! -e "$p7_dir"

git clone "$bundle" "$c2_dir"
git -C "$c2_dir" checkout --detach "$c2_tag"
test "$(git -C "$c2_dir" rev-parse HEAD)" = "$c2_commit"
mkdir -p "$c2_dir/datasets"
ln -s "$dataset" "$c2_dir/datasets/Covid19"

git clone "$bundle" "$p7_dir"
git -C "$p7_dir" checkout --detach "$p7_tag"
test "$(git -C "$p7_dir" rev-parse HEAD)" = "$p7_commit"
mkdir -p "$p7_dir/datasets"
ln -s "$dataset" "$p7_dir/datasets/Covid19"

for script in \
  "$c2_dir/scripts/run_cdrr_pair_server.sh" \
  "$p7_dir/scripts/run_cdrr_pair_server.sh" \
  "$p7_dir/scripts/start_cdrr_pair_server.sh"; do
  bash -n "$script"
done

printf 'C2_HEAD=%s\n' "$(git -C "$c2_dir" rev-parse HEAD)"
printf 'P7_HEAD=%s\n' "$(git -C "$p7_dir" rev-parse HEAD)"
printf 'C2_DIRTY=%s\n' "$(git -C "$c2_dir" status --porcelain --untracked-files=no | wc -l)"
printf 'P7_DIRTY=%s\n' "$(git -C "$p7_dir" status --porcelain --untracked-files=no | wc -l)"
readlink -f "$c2_dir/datasets/Covid19"
readlink -f "$p7_dir/datasets/Covid19"

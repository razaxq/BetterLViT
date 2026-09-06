#!/usr/bin/env bash
# Explicit frozen repositories only. No Test evaluation or automatic extension.
set -euo pipefail
c8_repo="${1:?C8 repository required}"
p10_repo="${2:?P10 repository required}"
output="${3:?New output directory required}"
python_bin=/root/autodl-tmp/envs/betterlvit-paper/bin/python
baseline="${4:?Frozen C4 validation JSON required}"
test -f "$baseline"
test ! -e "$output"
mkdir -p "$output"
cp "$baseline" "$output/c4_validation.json"
exec 9>/root/betterlvit_race_pe.lock
flock -n 9
trap 'printf "failed\n" > "$output/chain.status"' ERR
trap 'printf "interrupted\n" > "$output/chain.status"; exit 143' TERM INT
export HF_HOME=/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=1219
export BETTERLVIT_SEED=1219 BETTERLVIT_EPOCHS=80 BETTERLVIT_BATCH_SIZE=16
export BETTERLVIT_TRAIN_DROP_LAST=1 BETTERLVIT_NUM_WORKERS=4
export BETTERLVIT_DETERMINISTIC=1 BETTERLVIT_CUDNN_ENABLED=1 BETTERLVIT_VIS_FREQUENCY=100000
export AUTO_TEST_EVALUATE=0 AUTO_EVALUATE=0 TEST_SPLIT_ALLOWED=0
for arm in c8 p10; do
    if [ "$arm" = c8 ]; then
        repo="$c8_repo"; profile=c8_race_pe_v2_aux
    else
        repo="$p10_repo"; profile=p10_race_pe_v2
    fi
    export BETTERLVIT_EXPERIMENT="$profile"
    export BETTERLVIT_GIT_COMMIT
    BETTERLVIT_GIT_COMMIT=$(git -C "$repo" rev-parse HEAD)
    test -z "$(git -C "$repo" status --porcelain --untracked-files=no)"
    "$python_bin" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["profile"]==sys.argv[2] and d["epochs"]==80 and d["test_split_allowed"] is False' "$repo/experiment_manifests/active_race_pe_v2.json" "$profile"
    printf '%s %s\n' "$profile" "$BETTERLVIT_GIT_COMMIT" >> "$output/provenance.txt"
    printf '%s_training\n' "$arm" > "$output/chain.status"
    (cd "$repo" && "$python_bin" train_model.py) > "$output/${arm}_train.log" 2>&1
    printf '%s_exporting\n' "$arm" > "$output/chain.status"
    checkpoint=$(find "$repo/Covid19/BetterLViT/$profile" -type f -name 'best_model-BetterLViT.pth.tar' -print)
    test -f "$checkpoint"
    (cd "$repo" && "$python_bin" tools/export_validation_metrics.py --experiment "$profile" \
        --checkpoint "$checkpoint" --output "$output/${arm}_validation.json" --batch-size 16 --threshold 0.5) \
        > "$output/${arm}_export.log" 2>&1
done
printf 'comparing\n' > "$output/chain.status"
(cd "$p10_repo" && "$python_bin" tools/compare_race_pe.py --control "$output/c8_validation.json" \
    --candidate "$output/p10_validation.json" --output "$output/c8_vs_p10.json" --seed 1219) \
    > "$output/compare.log" 2>&1
for arm in c8 p10; do
    (cd "$p10_repo" && "$python_bin" tools/compare_race_pe.py --control "$output/c4_validation.json" \
        --candidate "$output/${arm}_validation.json" --output "$output/c4_vs_${arm}.json" --seed 1219) \
        > "$output/c4_vs_${arm}.log" 2>&1
done
printf 'complete\n' > "$output/chain.status"

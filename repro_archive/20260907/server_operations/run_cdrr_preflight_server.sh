#!/usr/bin/env bash
set -euo pipefail

python_bin=/root/autodl-tmp/envs/betterlvit-paper/bin/python
c2_dir=/root/autodl-tmp/BetterLViT-paper-c2-cdrr
p7_dir=/root/autodl-tmp/BetterLViT-paper-p7-cdrr
c2_commit=06479cd3302a8ca11022eac0a6b62bdad097eb65
p7_commit=fe4547a0c60fe948c9a574d9afc7d691370aeb42
output_dir=/root/cdrr_formal_preflight

export HF_HOME=/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=1219
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$output_dir"

test "$(git -C "$c2_dir" rev-parse HEAD)" = "$c2_commit"
test "$(git -C "$p7_dir" rev-parse HEAD)" = "$p7_commit"
test -z "$(git -C "$c2_dir" status --porcelain --untracked-files=no)"
test -z "$(git -C "$p7_dir" status --porcelain --untracked-files=no)"

cd "$p7_dir"
"$python_bin" tools/check_cdrr.py | tee "$output_dir/cdrr_module_check.txt"

for run in 1 2; do
  cd "$c2_dir"
  "$python_bin" tools/smoke_paper_profile.py \
    --experiment c2_cdrr_control --batch-size 16 --seed 1219 \
    | tee "$output_dir/c2_run${run}.json"
  cd "$p7_dir"
  "$python_bin" tools/smoke_paper_profile.py \
    --experiment p7_cdrr_v1 --batch-size 16 --seed 1219 \
    | tee "$output_dir/p7_run${run}.json"
done

"$python_bin" - "$output_dir" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])

def load(path):
    text = path.read_text(encoding="utf-8")
    start = text.find("{")
    if start < 0:
        raise RuntimeError(f"JSON not found in {path}")
    return json.loads(text[start:])

c2 = [load(root / f"c2_run{i}.json") for i in (1, 2)]
p7 = [load(root / f"p7_run{i}.json") for i in (1, 2)]
for pair in (c2, p7):
    assert pair[0]["output_sha256"] == pair[1]["output_sha256"]
    assert pair[0]["loss"] == pair[1]["loss"]
assert c2[0]["output_sha256"] == p7[0]["output_sha256"]
assert p7[0]["cdrr_identity_max_abs_error"] == 0.0
for result in c2 + p7:
    assert result["text_use_lora"] is False
    assert result["boundary_loss_weight"] == 0.0
print(json.dumps({
    "c2_output_sha256": c2[0]["output_sha256"],
    "p7_output_sha256": p7[0]["output_sha256"],
    "c2_loss": c2[0]["loss"],
    "p7_loss": p7[0]["loss"],
    "p7_identity_max_abs_error": p7[0]["cdrr_identity_max_abs_error"],
    "all_checks_passed": True,
}, indent=2, sort_keys=True))
PY

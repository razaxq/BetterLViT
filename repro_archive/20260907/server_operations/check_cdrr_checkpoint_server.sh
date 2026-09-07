#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/BetterLViT-paper-c2-cdrr
expected=06479cd3302a8ca11022eac0a6b62bdad097eb65
python_bin=/root/autodl-tmp/envs/betterlvit-paper/bin/python
session=$(find "$repo/Covid19/BetterLViT/c2_cdrr_control" -mindepth 1 -maxdepth 1 -type d -printf '%T@|%p\n' | sort -n | tail -1 | cut -d'|' -f2-)
test -n "$session"

"$python_bin" - "$session" "$expected" <<'PY'
import json
import pathlib
import sys
import torch

session = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
result = {"session": str(session), "checkpoints": {}}
for name in ("last_model-BetterLViT.pth.tar", "best_model-BetterLViT.pth.tar"):
    path = session / "models" / name
    if not path.is_file():
        if name.startswith("best_"):
            result["checkpoints"][name] = {"present": False, "early_training_allowed": True}
            continue
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    commit = checkpoint.get("source_git_commit")
    if commit != expected:
        raise AssertionError(f"{name}: source_git_commit={commit!r}")
    if checkpoint.get("text_use_lora") is not False:
        raise AssertionError(f"{name}: text_use_lora is not false")
    if float(checkpoint.get("boundary_loss_weight", -1.0)) != 0.0:
        raise AssertionError(f"{name}: boundary_loss_weight is not zero")
    if checkpoint.get("cdrr_enabled") is not False:
        raise AssertionError(f"{name}: C2 unexpectedly enables CDRR")
    result["checkpoints"][name] = {
        "bytes": path.stat().st_size,
        "epoch": checkpoint.get("epoch"),
        "source_git_commit": commit,
        "text_use_lora": checkpoint.get("text_use_lora"),
        "boundary_loss_weight": checkpoint.get("boundary_loss_weight"),
        "cdrr_enabled": checkpoint.get("cdrr_enabled"),
        "loadable": True,
    }
print(json.dumps(result, indent=2, sort_keys=True))
PY

grep -Ein 'nan|inf|cuda.*out of memory|out of memory|traceback|runtimeerror|error:' "$session"/*.log || true
df -h /root/autodl-tmp

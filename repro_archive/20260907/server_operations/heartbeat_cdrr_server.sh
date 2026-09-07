#!/usr/bin/env bash
set -euo pipefail

metadata=/root/autodl-tmp/BetterLViT-paper-p7-cdrr/runtime_logs/cdrr_pair_current.env
source "$metadata"
printf 'STATUS=%s\n' "$(cat "$STATUS")"
printf 'C2_HEAD=%s\n' "$(git -C "$C2_REPO" rev-parse HEAD)"
printf 'P7_HEAD=%s\n' "$(git -C "$P7_REPO" rev-parse HEAD)"
printf 'SHARED_BYTES=%s\n' "$(du -sb /root/autodl-fs/betterlvit_5090_migration | cut -f1)"
df -h /root/autodl-tmp /root/autodl-fs
pgrep -af '[t]rain_model.py|[e]xport_validation_metrics.py|[r]un_cdrr_pair_server.sh' || true
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader

for name in c2 p7; do
  log="$OUTPUT_DIR/${name}_train.stderr.log"
  if [ -f "$log" ]; then
    printf '%s_PROGRESS\n' "${name^^}"
    grep -E '========= Epoch|\[Train\].*\[[0-9]+/[0-9]+\]|\[Val\].*\[[0-9]+/[0-9]+\]|CDRR:' "$log" | tail -18 || true
    printf '%s_ERRORS\n' "${name^^}"
    grep -Ein '(^|[^a-z])(nan|inf)([^a-z]|$)|cuda.*out of memory|out of memory|traceback|runtimeerror|error:' "$log" | tail -20 || true
  fi
done

python_bin=/root/autodl-tmp/envs/betterlvit-paper/bin/python
"$python_bin" - "$OUTPUT_DIR" "$C2_REPO" "$P7_REPO" <<'PY'
import json
import pathlib
import sys
import torch

output = pathlib.Path(sys.argv[1])
repos = {
    "c2": (pathlib.Path(sys.argv[2]), "c2_cdrr_control", "06479cd3302a8ca11022eac0a6b62bdad097eb65", False),
    "p7": (pathlib.Path(sys.argv[3]), "p7_cdrr_v1", "fe4547a0c60fe948c9a574d9afc7d691370aeb42", True),
}
summary = {}
for key, (repo, experiment, expected, cdrr_expected) in repos.items():
    sessions = sorted((repo / "Covid19" / "BetterLViT" / experiment).glob("*_session_*"), key=lambda p: p.stat().st_mtime)
    if not sessions:
        continue
    session = sessions[-1]
    checks = {}
    for filename in ("last_model-BetterLViT.pth.tar", "best_model-BetterLViT.pth.tar"):
        path = session / "models" / filename
        if not path.exists():
            checks[filename] = {"present": False}
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checks[filename] = {
            "present": True,
            "loadable": True,
            "epoch": checkpoint.get("epoch"),
            "source_git_commit": checkpoint.get("source_git_commit"),
            "text_use_lora": checkpoint.get("text_use_lora"),
            "boundary_loss_weight": checkpoint.get("boundary_loss_weight"),
            "cdrr_enabled": checkpoint.get("cdrr_enabled"),
        }
        assert checkpoint.get("source_git_commit") == expected
        assert checkpoint.get("text_use_lora") is False
        assert float(checkpoint.get("boundary_loss_weight", -1)) == 0.0
        assert checkpoint.get("cdrr_enabled") is cdrr_expected
        del checkpoint
    summary[key] = {"session": str(session), "checkpoints": checks}

c2_json = output / "c2_validation.json"
if c2_json.exists():
    data = json.loads(c2_json.read_text(encoding="utf-8"))
    summary["c2_validation"] = {
        "valid_json": True,
        "split": data.get("split"),
        "sample_count": data.get("sample_count"),
        "test_split_accessed": data.get("test_split_accessed"),
        "git_commit": data.get("git_commit"),
        "metrics": data.get("metrics"),
    }
print(json.dumps(summary, indent=2, sort_keys=True))
PY

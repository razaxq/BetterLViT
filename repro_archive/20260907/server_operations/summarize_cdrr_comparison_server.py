import json
from pathlib import Path

root = Path("/root/autodl-tmp/BetterLViT-paper-p7-cdrr/runtime_logs/cdrr_pair_20260903_122723")
c2 = json.loads((root / "c2_validation.json").read_text(encoding="utf-8"))
p7 = json.loads((root / "p7_validation.json").read_text(encoding="utf-8"))
comparison = json.loads((root / "c2_vs_p7.json").read_text(encoding="utf-8"))

assert c2["split"] == p7["split"] == comparison["split"] == "validation"
assert c2["test_split_accessed"] is p7["test_split_accessed"] is comparison["test_split_accessed"] is False
assert c2["samples"] == p7["samples"] == comparison["samples"] == 1429
assert c2["checkpoint_git_commit"] == "06479cd3302a8ca11022eac0a6b62bdad097eb65"
assert p7["checkpoint_git_commit"] == "fe4547a0c60fe948c9a574d9afc7d691370aeb42"
assert p7["text_use_lora"] is False and p7["boundary_loss_weight"] == 0.0 and p7["cdrr_enabled"] is True

summary = {
    "c2": {key: c2[key] for key in (
        "checkpoint_best_epoch", "macro_dice", "macro_iou", "macro_precision",
        "macro_recall", "macro_brier", "macro_boundary_f1_tolerance_2")},
    "p7": {key: p7[key] for key in (
        "checkpoint_best_epoch", "macro_dice", "macro_iou", "macro_precision",
        "macro_recall", "macro_brier", "macro_boundary_f1_tolerance_2")},
    "overall_delta": {
        key: comparison["overall"][key]
        for key in ("dice", "iou", "precision", "recall", "boundary_f1_tolerance_2", "brier_lower_is_better")
    },
    "smallest_lesion_quartile": comparison["lesion_size_quartiles"][0],
    "highest_hf_laplacian_quartile": comparison["image_frequency_quartiles"]["hf_laplacian_energy"][-1],
    "highest_normalized_detail_quartile": comparison["image_frequency_quartiles"]["hf_normalized_local_detail"][-1],
    "high_frequency_gate": comparison["high_frequency_gate"],
    "passes_numeric_screen": comparison["passes_numeric_screen"],
    "cdrr_stats": p7["cdrr_stats_last_batch"],
}
print(json.dumps(summary, indent=2, sort_keys=True))

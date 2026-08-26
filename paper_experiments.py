"""Pre-registered experiment profiles for the dissertation ablation.

Only fields that are allowed to differ between profiles live here. Training
schedule, data split, augmentation, optimizer and evaluation stay centralized
in ``Config.py`` so an experiment cannot silently change unrelated settings.
"""


PAPER_EXPERIMENTS = {
    "b0_baseline": {
        "paper_id": "B0",
        "description": "Frozen CXR-BERT + original PLAM + Dice/BCE",
        "decoder_fusion_mode": "legacy_plam",
        "text_use_lora": False,
        "loss_name": "dice_bce",
        "architecture_version": "paper_b0_cxrbert_plam_dice_bce",
    },
    "a0_lora": {
        "paper_id": "A0",
        "description": "CXR-BERT LoRA + original PLAM + Dice/BCE",
        "decoder_fusion_mode": "legacy_plam",
        "text_use_lora": True,
        "loss_name": "dice_bce",
        "architecture_version": "paper_a0_lora_plam_dice_bce",
    },
    "a1_lora_focal": {
        "paper_id": "A1",
        "description": "CXR-BERT LoRA + original PLAM + Dice/Focal",
        "decoder_fusion_mode": "legacy_plam",
        "text_use_lora": True,
        "loss_name": "dice_focal",
        "architecture_version": "paper_a1_lora_plam_dice_focal",
    },
    "a2_lora_freq": {
        "paper_id": "A2",
        "description": "CXR-BERT LoRA + FAM-EPPA V4-B frequency fusion + Dice/BCE",
        "decoder_fusion_mode": "fam_eppa_v4b",
        "text_use_lora": True,
        "loss_name": "dice_bce",
        "architecture_version": "paper_a2_lora_fam_eppa_v4b_dice_bce",
    },
    "a3_lora_fmiseg": {
        "paper_id": "A3",
        "description": "CXR-BERT LoRA + FMISeg-adapted decoder fusion + Dice/BCE",
        "decoder_fusion_mode": "fmiseg_adapter",
        "text_use_lora": True,
        "loss_name": "dice_bce",
        "architecture_version": "paper_a3_lora_fmiseg_adapter_dice_bce",
    },
}


def get_paper_experiment(name):
    """Return a copied, validated experiment profile."""
    normalized = str(name).strip().lower()
    if normalized not in PAPER_EXPERIMENTS:
        available = ", ".join(PAPER_EXPERIMENTS)
        raise ValueError(
            "Unknown BETTERLVIT_EXPERIMENT={!r}. Available: {}".format(
                name,
                available,
            )
        )
    profile = dict(PAPER_EXPERIMENTS[normalized])
    profile["name"] = normalized
    return profile

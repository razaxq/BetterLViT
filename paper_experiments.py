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
        "tcsr_enabled": False,
        "text_use_lora": False,
        "loss_name": "dice_bce",
        "architecture_version": "paper_b0_cxrbert_plam_dice_bce",
    },
    "a0_lora": {
        "paper_id": "A0",
        "description": "CXR-BERT LoRA + original PLAM + Dice/BCE",
        "decoder_fusion_mode": "legacy_plam",
        "tcsr_enabled": False,
        "text_use_lora": True,
        "loss_name": "dice_bce",
        "architecture_version": "paper_a0_lora_plam_dice_bce",
    },
    "a1_lora_focal": {
        "paper_id": "A1",
        "description": "CXR-BERT LoRA + original PLAM + Dice/Focal",
        "decoder_fusion_mode": "legacy_plam",
        "tcsr_enabled": False,
        "text_use_lora": True,
        "loss_name": "dice_focal",
        "architecture_version": "paper_a1_lora_plam_dice_focal",
    },
    "a2_lora_freq": {
        "paper_id": "A2",
        "description": "CXR-BERT LoRA + FAM-EPPA V4-B frequency fusion + Dice/BCE",
        "decoder_fusion_mode": "fam_eppa_v4b",
        "tcsr_enabled": False,
        "text_use_lora": True,
        "loss_name": "dice_bce",
        "architecture_version": "paper_a2_lora_fam_eppa_v4b_dice_bce",
    },
    "a3_lora_fmiseg": {
        "paper_id": "A3",
        "description": "CXR-BERT LoRA + FMISeg-adapted decoder fusion + Dice/BCE",
        "decoder_fusion_mode": "fmiseg_adapter",
        "tcsr_enabled": False,
        "text_use_lora": True,
        "loss_name": "dice_bce",
        "architecture_version": "paper_a3_lora_fmiseg_adapter_dice_bce",
    },
    "a4_lora_freq_focal": {
        "paper_id": "A4",
        "description": "CXR-BERT LoRA + FAM-EPPA V4-B frequency fusion + Dice/Focal",
        "decoder_fusion_mode": "fam_eppa_v4b",
        "tcsr_enabled": False,
        "text_use_lora": True,
        "loss_name": "dice_focal",
        "architecture_version": "paper_a4_lora_fam_eppa_v4b_dice_focal",
    },
    "a6_tcsr": {
        "paper_id": "A6",
        "description": "Frozen CXR-BERT + TCSR + original PLAM + Dice/BCE",
        "decoder_fusion_mode": "legacy_plam",
        "tcsr_enabled": True,
        "text_use_lora": False,
        "loss_name": "dice_bce",
        "architecture_version": "paper_a6_frozen_tcsr_v1_plam_dice_bce",
    },
    "a7_tcsr_freq": {
        "paper_id": "A7",
        "description": "Frozen CXR-BERT + TCSR + FAM-EPPA V4-B + Dice/BCE",
        "decoder_fusion_mode": "fam_eppa_v4b",
        "tcsr_enabled": True,
        "text_use_lora": False,
        "loss_name": "dice_bce",
        "architecture_version": "paper_a7_frozen_tcsr_v1_fam_eppa_v4b_dice_bce",
    },
    "a8_tcsr_freq_focal": {
        "paper_id": "A8",
        "description": "Frozen CXR-BERT + TCSR + FAM-EPPA V4-B + Dice/Focal",
        "decoder_fusion_mode": "fam_eppa_v4b",
        "tcsr_enabled": True,
        "text_use_lora": False,
        "loss_name": "dice_focal",
        "architecture_version": "paper_a8_frozen_tcsr_v1_fam_eppa_v4b_dice_focal",
    },
    "a9_frozen_freq_focal": {
        "paper_id": "A9",
        "description": "Frozen CXR-BERT + FAM-EPPA V4-B frequency fusion + Dice/Focal",
        "decoder_fusion_mode": "fam_eppa_v4b",
        "text_use_lora": False,
        "loss_name": "dice_focal",
        "architecture_version": "paper_a9_frozen_fam_eppa_v4b_dice_focal",
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

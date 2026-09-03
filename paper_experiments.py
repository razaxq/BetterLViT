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
    "a4_lora_freq_focal": {
        "paper_id": "A4",
        "description": "CXR-BERT LoRA + FAM-EPPA V4-B frequency fusion + Dice/Focal",
        "decoder_fusion_mode": "fam_eppa_v4b",
        "text_use_lora": True,
        "loss_name": "dice_focal",
        "architecture_version": "paper_a4_lora_fam_eppa_v4b_dice_focal",
    },
    "a9_frozen_freq_focal": {
        "paper_id": "A9",
        "description": "Frozen CXR-BERT + FAM-EPPA V4-B frequency fusion + Dice/Focal",
        "decoder_fusion_mode": "fam_eppa_v4b",
        "text_use_lora": False,
        "loss_name": "dice_focal",
        "architecture_version": "paper_a9_frozen_fam_eppa_v4b_dice_focal",
    },
    "c1_bcdh_control": {
        "paper_id": "C1",
        "description": (
            "Frozen CXR-BERT + FAM-EPPA V4-B + Dice/Focal; "
            "BCDH validation control"
        ),
        "decoder_fusion_mode": "fam_eppa_v4b",
        "text_use_lora": False,
        "loss_name": "dice_focal",
        "bcdh_enabled": False,
        "architecture_version": (
            "pilot_c1_frozen_fam_eppa_v4b_dice_focal_control"
        ),
    },
    "p6_bcdh_r_v1": {
        "paper_id": "P6",
        "description": (
            "Frozen CXR-BERT + FAM-EPPA V4-B + BCDH-R V1 + Dice/Focal"
        ),
        "decoder_fusion_mode": "fam_eppa_v4b",
        "text_use_lora": False,
        "loss_name": "dice_focal",
        "bcdh_enabled": True,
        "bcdh_aux_weight": 0.2,
        "bcdh_hidden_channels": 32,
        "bcdh_delta_max": 1.0,
        "bcdh_detach_cues": True,
        "architecture_version": (
            "pilot_p6_frozen_fam_eppa_v4b_bcdh_r_v1_dice_focal"
        ),
    },
    "c2_cdrr_control": {
        "paper_id": "C2",
        "description": (
            "Frozen CXR-BERT + FAM-EPPA V4-B + Dice/Focal; "
            "CDRR validation control"
        ),
        "decoder_fusion_mode": "fam_eppa_v4b",
        "text_use_lora": False,
        "loss_name": "dice_focal",
        "bcdh_enabled": False,
        "cdrr_enabled": False,
        "architecture_version": (
            "pilot_c2_frozen_fam_eppa_v4b_dice_focal_control"
        ),
    },
    "p7_cdrr_v1": {
        "paper_id": "P7",
        "description": (
            "Frozen CXR-BERT + FAM-EPPA V4-B + CDRR V1 + Dice/Focal"
        ),
        "decoder_fusion_mode": "fam_eppa_v4b",
        "text_use_lora": False,
        "loss_name": "dice_focal",
        "bcdh_enabled": False,
        "cdrr_enabled": True,
        "cdrr_aux_weight": 0.1,
        "cdrr_hidden_channels": 32,
        "cdrr_delta_max": 0.5,
        "cdrr_active_fraction": 0.15,
        "architecture_version": (
            "pilot_p7_frozen_fam_eppa_v4b_cdrr_v1_dice_focal"
        ),
    },
    "c3_race_control": {
        "paper_id": "C3",
        "description": (
            "Frozen CXR-BERT + FAM-EPPA V4-B + Dice/Focal; "
            "RACE-Fuse validation control"
        ),
        "decoder_fusion_mode": "fam_eppa_v4b",
        "text_use_lora": False,
        "loss_name": "dice_focal",
        "bcdh_enabled": False,
        "cdrr_enabled": False,
        "race_enabled": False,
        "architecture_version": (
            "pilot_c3_frozen_fam_eppa_v4b_dice_focal_control"
        ),
    },
    "p8_race_fuse_v1": {
        "paper_id": "P8",
        "description": (
            "Frozen CXR-BERT + RACE-Fuse V1 + FAM-EPPA V4-B + "
            "Dice/Focal"
        ),
        "decoder_fusion_mode": "fam_eppa_v4b",
        "text_use_lora": False,
        "loss_name": "dice_focal",
        "bcdh_enabled": False,
        "cdrr_enabled": False,
        "race_enabled": True,
        "race_aux_weight": 0.05,
        "race_hidden_channels": 32,
        "race_max_strength": 0.15,
        "architecture_version": (
            "pilot_p8_frozen_race_fuse_v1_fam_eppa_v4b_dice_focal"
        ),
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

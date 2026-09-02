"""Export per-image validation metrics without touching the Test split."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_profile_parser = argparse.ArgumentParser(add_help=False)
_profile_parser.add_argument("--experiment")
_profile_args, _ = _profile_parser.parse_known_args()
if _profile_args.experiment:
    os.environ["BETTERLVIT_EXPERIMENT"] = _profile_args.experiment

import Config as config
from Load_Dataset import ImageToImage2D, ValGenerator
from nets.BetterLViT import BetterLViT
from utils import read_text


ALLOWED_EXPERIMENTS = (
    "a9_frozen_freq_focal",
    "p1_tcsrv21_boundary_router",
    "p2_tcsrv22_single_hop_boundary",
    "p3_tcsrv23_calibrated_gate",
    "p4_tcsrv24_sparse_boundary",
    "c0_frozen_freq_tversky",
    "p5_tcsrv25_local_tversky",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        required=True,
        choices=ALLOWED_EXPERIMENTS,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def build_model():
    return BetterLViT(
        config.get_CTranS_config(),
        n_channels=config.n_channels,
        n_classes=config.n_labels,
        text_encoder_name=config.text_encoder_name,
        text_seq_len=config.text_max_len,
        use_lora=config.text_use_lora,
        lora_r=config.text_lora_r,
        lora_alpha=config.text_lora_alpha,
        lora_dropout=config.text_lora_dropout,
        lora_target_modules=config.text_lora_target_modules,
    )


def validation_loader(batch_size):
    text = read_text(os.path.join(
        config.task_dataset,
        "Train_Val_text.xlsx",
    ))
    dataset = ImageToImage2D(
        config.val_dataset,
        config.task_name,
        text,
        ValGenerator(output_size=[config.img_size, config.img_size]),
        image_size=config.img_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return dataset, loader


def git_commit():
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def main():
    args = parse_args()
    if args.batch_size <= 0 or not 0.0 <= args.threshold <= 1.0:
        raise ValueError("Invalid batch size or threshold.")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-capable environment is required.")

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if checkpoint.get("experiment_name") != config.experiment_name:
        raise RuntimeError(
            "Checkpoint experiment mismatch: expected {!r}, found {!r}."
            .format(
                config.experiment_name,
                checkpoint.get("experiment_name"),
            )
        )
    expected_architecture = config.experiment_architecture_version
    if checkpoint.get("architecture_version") != expected_architecture:
        raise RuntimeError(
            "Checkpoint architecture mismatch: expected {!r}, found {!r}."
            .format(
                expected_architecture,
                checkpoint.get("architecture_version"),
            )
        )

    torch.backends.cudnn.enabled = config.cudnn_enabled
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = config.deterministic_training
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(config.deterministic_training)

    model = build_model()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.cuda().eval()
    dataset, loader = validation_loader(args.batch_size)

    records = []
    with torch.inference_mode():
        for batch, names in tqdm(
            loader,
            desc="Validation-only",
            unit="batch",
            ncols=80,
        ):
            probabilities = model(
                batch["image"].cuda(non_blocking=True),
                batch["input_ids"].cuda(non_blocking=True),
                batch["attention_mask"].cuda(non_blocking=True),
            )[:, 0].float().cpu()
            labels = batch["label"].bool()
            if labels.ndim != 3:
                raise RuntimeError(
                    "Expected validation labels [B, H, W], received {}."
                    .format(tuple(labels.shape))
                )
            predictions = probabilities > args.threshold
            intersection = (predictions & labels).sum(
                dim=(1, 2),
                dtype=torch.float64,
            )
            prediction_sum = predictions.sum(
                dim=(1, 2),
                dtype=torch.float64,
            )
            label_sum = labels.sum(
                dim=(1, 2),
                dtype=torch.float64,
            )
            denominator = prediction_sum + label_sum
            union = denominator - intersection
            dice = torch.where(
                denominator > 0,
                2.0 * intersection / denominator,
                torch.zeros_like(intersection),
            )
            iou = torch.where(
                union > 0,
                intersection / union,
                torch.zeros_like(intersection),
            )
            for index, name in enumerate(names):
                records.append({
                    "name": str(name),
                    "label_pixels": int(label_sum[index].item()),
                    "prediction_pixels": int(prediction_sum[index].item()),
                    "dice": float(dice[index].item()),
                    "iou": float(iou[index].item()),
                })

    if len(records) != len(dataset):
        raise RuntimeError(
            "Validation sample mismatch: {} != {}.".format(
                len(records),
                len(dataset),
            )
        )
    dice_values = np.asarray([record["dice"] for record in records])
    iou_values = np.asarray([record["iou"] for record in records])
    if not np.isfinite(dice_values).all() or not np.isfinite(iou_values).all():
        raise RuntimeError("Validation metrics contain non-finite values.")

    router = getattr(model, "tcsr", None)
    result = {
        "split": "validation",
        "test_split_accessed": False,
        "experiment": config.experiment_name,
        "paper_id": config.experiment_paper_id,
        "architecture_version": config.experiment_architecture_version,
        "checkpoint": str(checkpoint_path),
        "checkpoint_git_commit": checkpoint.get("source_git_commit"),
        "analysis_git_commit": git_commit(),
        "checkpoint_best_epoch": int(checkpoint.get("best_epoch", -1)),
        "threshold": float(args.threshold),
        "samples": len(records),
        "macro_dice": float(dice_values.mean()),
        "macro_iou": float(iou_values.mean()),
        "text_use_lora": bool(config.text_use_lora),
        "tcsr_enabled": bool(config.tcsr_enabled),
        "tcsr_version": config.tcsr_version,
        "tcsr_stats_last_batch": dict(
            getattr(router, "_last_stats", {}) or {}
        ),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps({
        key: value
        for key, value in result.items()
        if key != "records"
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

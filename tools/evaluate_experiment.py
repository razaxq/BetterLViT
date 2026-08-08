"""Select a threshold on validation, then evaluate the full test split."""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import Config as config
from Load_Dataset import ImageToImage2D, ValGenerator
from nets.BetterLViT import BetterLViT
from utils import read_text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--minimum", type=float, default=0.30)
    parser.add_argument("--maximum", type=float, default=0.70)
    parser.add_argument("--step", type=float, default=0.002)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save binary test masks as PNG files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Prediction output directory. Defaults to a threshold-named "
            "folder beside the checkpoint session."
        ),
    )
    parser.add_argument(
        "--prediction-threshold",
        type=float,
        help=(
            "Threshold used only for saved masks. Defaults to the threshold "
            "selected on validation."
        ),
    )
    return parser.parse_args()


def latest_best_checkpoint():
    candidates = list(
        (REPO_ROOT / config.task_name / config.model_name).glob(
            f"*/models/best_model-{config.model_name}.pth.tar"
        )
    )
    if not candidates:
        raise FileNotFoundError("No BetterLViT experiment checkpoint found.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


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


def make_loader(path, text_file, text_root=None, batch_size=16):
    text_root = path if text_root is None else text_root
    text = read_text(os.path.join(text_root, text_file))
    dataset = ImageToImage2D(
        path,
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


def accumulate_metrics(probabilities, labels, thresholds):
    predictions = (
        probabilities[:, None]
        > thresholds[None, :, None, None]
    )
    labels = labels[:, None]
    intersection = (
        predictions & labels
    ).sum(dim=(2, 3), dtype=torch.float64)
    prediction_sum = predictions.sum(
        dim=(2, 3), dtype=torch.float64
    )
    label_sum = labels.sum(dim=(2, 3), dtype=torch.float64)
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
    return dice.sum(dim=0), iou.sum(dim=0)


@torch.inference_mode()
def evaluate_thresholds(
    model,
    loader,
    sample_count,
    thresholds,
    description,
    prediction_output_dir=None,
    prediction_threshold=None,
):
    threshold_cpu = torch.as_tensor(
        thresholds,
        dtype=torch.float32,
        device="cpu",
    )
    dice_sums = torch.zeros(
        len(thresholds),
        dtype=torch.float64,
        device="cpu",
    )
    iou_sums = torch.zeros_like(dice_sums)
    if prediction_output_dir is not None:
        if prediction_threshold is None:
            raise ValueError(
                "prediction_threshold is required when saving predictions."
            )
        prediction_output_dir.mkdir(parents=True, exist_ok=True)

    saved_prediction_count = 0
    for batch, names in tqdm(
        loader,
        desc=description,
        unit="batch",
        ncols=80,
    ):
        probabilities = model(
            batch["image"].cuda(non_blocking=True),
            batch["input_ids"].cuda(non_blocking=True),
            batch["attention_mask"].cuda(non_blocking=True),
        )[:, 0].float().cpu()
        dice, iou = accumulate_metrics(
            probabilities,
            batch["label"].bool(),
            threshold_cpu,
        )
        dice_sums += dice
        iou_sums += iou

        if prediction_output_dir is not None:
            prediction_masks = (
                probabilities > prediction_threshold
            ).numpy().astype(np.uint8) * 255
            for prediction_mask, image_name in zip(
                prediction_masks,
                names,
            ):
                output_name = (
                    f"{Path(image_name).stem}_pred.png"
                )
                output_path = prediction_output_dir / output_name
                if not cv2.imwrite(
                    str(output_path),
                    prediction_mask,
                ):
                    raise OSError(
                        f"Failed to save prediction: {output_path}"
                    )
                saved_prediction_count += 1

    if (
        prediction_output_dir is not None
        and saved_prediction_count != sample_count
    ):
        raise RuntimeError(
            "Prediction count mismatch: "
            f"{saved_prediction_count} != {sample_count}"
        )

    dice_means = (dice_sums / sample_count).numpy()
    iou_means = (iou_sums / sample_count).numpy()
    if (
        not np.isfinite(dice_means).all()
        or not np.isfinite(iou_means).all()
        or (dice_means < 0).any()
        or (dice_means > 1).any()
        or (iou_means < 0).any()
        or (iou_means > 1).any()
        or (iou_means > dice_means + 1e-12).any()
    ):
        raise RuntimeError(
            "Invalid segmentation metrics; refusing to save results."
        )
    return dice_means, iou_means


def main():
    args = parse_args()
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("An AMD ROCm PyTorch environment is required.")
    if (
        args.step <= 0
        or args.maximum < args.minimum
        or args.batch_size <= 0
        or (
            args.prediction_threshold is not None
            and not 0.0 <= args.prediction_threshold <= 1.0
        )
    ):
        raise ValueError("Invalid evaluation arguments.")

    checkpoint_path = (
        args.checkpoint.resolve()
        if args.checkpoint
        else latest_best_checkpoint().resolve()
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    expected_architecture = getattr(
        config,
        "experiment_architecture_version",
        None,
    )
    checkpoint_architecture = checkpoint.get("architecture_version")
    if (
        getattr(config, "require_checkpoint_architecture_match", False)
        and checkpoint_architecture != expected_architecture
    ):
        raise RuntimeError(
            "Checkpoint architecture mismatch: expected {!r}, found {!r}".format(
                expected_architecture,
                checkpoint_architecture,
            )
        )
    model = build_model()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.cuda().eval()

    validation_dataset, validation_loader = make_loader(
        config.val_dataset,
        "Train_Val_text.xlsx",
        text_root=config.task_dataset,
        batch_size=args.batch_size,
    )
    thresholds = np.arange(
        args.minimum,
        args.maximum + args.step * 0.5,
        args.step,
    )
    val_dice, val_iou = evaluate_thresholds(
        model,
        validation_loader,
        len(validation_dataset),
        thresholds,
        "Calibrate",
    )
    best_index = int(np.argmax(val_dice))
    default_index = int(np.argmin(np.abs(thresholds - 0.5)))
    selected_threshold = float(thresholds[best_index])

    test_dataset, test_loader = make_loader(
        config.test_dataset,
        "Test_text.xlsx",
        batch_size=args.batch_size,
    )
    prediction_output_dir = None
    prediction_threshold = (
        args.prediction_threshold
        if args.prediction_threshold is not None
        else selected_threshold
    )
    if args.save_predictions:
        prediction_output_dir = (
            args.output_dir.resolve()
            if args.output_dir
            else (
                checkpoint_path.parent.parent
                / (
                    "test_predictions_threshold_"
                    f"{prediction_threshold:.3f}"
                )
            )
        )
    test_thresholds = np.asarray([0.5, selected_threshold])
    test_dice, test_iou = evaluate_thresholds(
        model,
        test_loader,
        len(test_dataset),
        test_thresholds,
        "Test",
        prediction_output_dir=prediction_output_dir,
        prediction_threshold=prediction_threshold,
    )

    baselines = {
        "839752": {
            "threshold_0_5": {
                "dice": 0.837091,
                "iou": 0.750527,
            },
            "validation_selected": {
                "threshold": 0.58,
                "dice": 0.839025,
                "iou": 0.754068,
            },
        },
        "boundary_loss": {
            "threshold_0_5": {
                "dice": 0.841424,
                "iou": 0.756196,
            },
            "validation_selected": {
                "threshold": 0.608,
                "dice": 0.842278,
                "iou": 0.758155,
            },
        },
        "dg_eppa_v1": {
            "threshold_0_5": {
                "dice": 0.841424,
                "iou": 0.755901,
            },
            "validation_selected": {
                "threshold": 0.512,
                "dice": 0.841512,
                "iou": 0.756163,
            },
        },
        "br_dg_eppa_v2": {
            "threshold_0_5": {
                "dice": 0.8363975942502839,
                "iou": 0.7501020239160945,
            },
            "validation_selected": {
                "threshold": 0.542,
                "dice": 0.8370015853060114,
                "iou": 0.7514141235748568,
            },
        },
    }
    result = {
        "checkpoint": str(checkpoint_path),
        "architecture": checkpoint.get(
            "architecture",
            getattr(config, "experiment_architecture", "EPPA"),
        ),
        "architecture_version": checkpoint_architecture,
        "best_epoch": int(checkpoint.get("best_epoch", -1)),
        "boundary_loss_weight": config.boundary_loss_weight,
        "loss_name": getattr(config, "loss_name", "dice_bce"),
        "validation": {
            "samples": len(validation_dataset),
            "threshold_0_5": {
                "dice": float(val_dice[default_index]),
                "iou": float(val_iou[default_index]),
            },
            "selected": {
                "threshold": selected_threshold,
                "dice": float(val_dice[best_index]),
                "iou": float(val_iou[best_index]),
            },
        },
        "test": {
            "samples": len(test_dataset),
            "prediction_directory": (
                str(prediction_output_dir)
                if prediction_output_dir
                else None
            ),
            "prediction_threshold": (
                prediction_threshold
                if prediction_output_dir
                else None
            ),
            "threshold_0_5": {
                "dice": float(test_dice[0]),
                "iou": float(test_iou[0]),
            },
            "validation_selected_threshold": {
                "threshold": selected_threshold,
                "dice": float(test_dice[1]),
                "iou": float(test_iou[1]),
            },
        },
        "baselines_to_beat": baselines,
        "calibrated_delta": {
            name: {
                "dice": float(
                    test_dice[1]
                    - values["validation_selected"]["dice"]
                ),
                "iou": float(
                    test_iou[1]
                    - values["validation_selected"]["iou"]
                ),
            }
            for name, values in baselines.items()
        },
    }
    output_path = (
        checkpoint_path.parent.parent
        / getattr(
            config,
            "experiment_output_name",
            "experiment_evaluation.json",
        )
    )
    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    if prediction_output_dir is not None:
        manifest_path = prediction_output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint_path),
                    "threshold": prediction_threshold,
                    "validation_selected_threshold": selected_threshold,
                    "samples": len(test_dataset),
                    "file_pattern": "*_pred.png",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == "nt":
        os._exit(0)

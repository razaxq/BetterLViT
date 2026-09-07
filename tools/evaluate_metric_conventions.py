"""Report BOTH metric conventions on the same predictions.

Why this exists
---------------
LViT's official ``test_model.py`` computes a per-image Dice/IoU and averages
over the 2,113 test images (macro / per-image mean).  This repository's
``tools/evaluate_experiment.py`` uses the same convention, which is why our
reproduction (0.8371) matches the published LViT-T number (0.8366).

LanGuideMedSeg (MICCAI 2023) and SGSeg (MICCAI 2024) instead use
``torchmetrics.Dice()`` and ``torchmetrics.classification.BinaryJaccardIndex``
accumulated by a Lightning ``trainer.test()`` loop.  Those are *micro* metrics:
one global confusion matrix over every pixel of every image.

Micro Dice is systematically higher than macro Dice whenever a dataset contains
many small lesions, because small-lesion images -- which score poorly per image
-- contribute almost no pixels to the global count.  On the QaTa-COV19-v2 test
split, 545 / 2113 images have a lesion covering under 5% of the frame, and the
smaller half of the images together hold only 21% of all lesion pixels.

So a paper reporting "Dice 89.78" and a paper reporting "Dice 83.66" may be
describing the *same* segmentation quality.  This script measures both on our
own checkpoint so the comparison can finally be made on one scale.

Usage
-----
    python tools/evaluate_metric_conventions.py --checkpoint <path/to/best.pth.tar>

Read-only: loads a checkpoint, runs inference, writes one JSON. It does not
train, does not touch existing checkpoints, and does not select anything on the
test split -- the operating threshold still comes from validation.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import Config as config  # noqa: E402
from Load_Dataset import ImageToImage2D, ValGenerator  # noqa: E402
from nets.BetterLViT import BetterLViT  # noqa: E402
from utils import read_text  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--minimum", type=float, default=0.30)
    parser.add_argument("--maximum", type=float, default=0.70)
    parser.add_argument("--step", type=float, default=0.002)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("metric_convention_report.json"),
    )
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


def make_loader(path, text_file, batch_size, text_root=None):
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


@torch.inference_mode()
def sweep(model, loader, thresholds, description):
    """Accumulate macro and micro statistics for every threshold at once."""
    n_thresh = len(thresholds)
    th = torch.as_tensor(thresholds, dtype=torch.float32).view(1, -1, 1, 1)

    macro_dice = torch.zeros(n_thresh, dtype=torch.float64)
    macro_iou = torch.zeros(n_thresh, dtype=torch.float64)
    micro_inter = torch.zeros(n_thresh, dtype=torch.float64)
    micro_pred = torch.zeros(n_thresh, dtype=torch.float64)
    micro_label = torch.zeros(n_thresh, dtype=torch.float64)
    samples = 0

    # Per-image Dice at the default 0.5 threshold, kept alongside lesion area
    # so the macro/micro gap can be attributed to lesion size.
    per_image_dice_at_half = []
    per_image_area = []
    half_index = int(np.argmin(np.abs(np.asarray(thresholds) - 0.5)))

    for batch, _names in tqdm(loader, desc=description, ncols=80):
        images = batch["image"].cuda(non_blocking=True)
        labels = batch["label"].cuda(non_blocking=True)
        input_ids = batch["input_ids"].cuda(non_blocking=True)
        attention_mask = batch["attention_mask"].cuda(non_blocking=True)

        logits = model(images, input_ids, attention_mask)
        probabilities = logits.squeeze(1).float().cpu()
        labels = labels.squeeze(1).bool().cpu() if labels.dim() == 4 else labels.bool().cpu()

        predictions = probabilities[:, None] > th
        target = labels[:, None]

        inter = (predictions & target).sum(dim=(2, 3), dtype=torch.float64)
        p_sum = predictions.sum(dim=(2, 3), dtype=torch.float64)
        l_sum = target.sum(dim=(2, 3), dtype=torch.float64)
        denom = p_sum + l_sum
        union = denom - inter

        dice = torch.where(denom > 0, 2.0 * inter / denom, torch.zeros_like(inter))
        iou = torch.where(union > 0, inter / union, torch.zeros_like(inter))

        macro_dice += dice.sum(dim=0)
        macro_iou += iou.sum(dim=0)
        micro_inter += inter.sum(dim=0)
        micro_pred += p_sum.sum(dim=0)
        micro_label += l_sum.sum(dim=0)
        samples += probabilities.shape[0]

        per_image_dice_at_half.append(dice[:, half_index].numpy())
        per_image_area.append(target[:, 0].sum(dim=(1, 2)).numpy())

    macro_dice /= samples
    macro_iou /= samples
    micro_denom = micro_pred + micro_label
    micro_union = micro_denom - micro_inter
    micro_dice = torch.where(
        micro_denom > 0, 2.0 * micro_inter / micro_denom, torch.zeros_like(micro_inter)
    )
    micro_iou = torch.where(
        micro_union > 0, micro_inter / micro_union, torch.zeros_like(micro_inter)
    )

    return {
        "samples": samples,
        "macro_dice": macro_dice.numpy(),
        "macro_iou": macro_iou.numpy(),
        "micro_dice": micro_dice.numpy(),
        "micro_iou": micro_iou.numpy(),
        "per_image_dice_at_0_5": np.concatenate(per_image_dice_at_half),
        "per_image_area": np.concatenate(per_image_area),
    }


def at(result, thresholds, threshold):
    index = int(np.argmin(np.abs(np.asarray(thresholds) - threshold)))
    return {
        "threshold": float(thresholds[index]),
        "macro_dice": float(result["macro_dice"][index]),
        "macro_iou": float(result["macro_iou"][index]),
        "micro_dice": float(result["micro_dice"][index]),
        "micro_iou": float(result["micro_iou"][index]),
    }


def size_breakdown(result):
    """Per-image Dice by lesion-size quartile: the mechanism behind the gap."""
    dice = result["per_image_dice_at_0_5"]
    area = result["per_image_area"].astype(np.float64)
    fraction = area / float(config.img_size * config.img_size)
    edges = np.percentile(fraction, [0, 25, 50, 75, 100])
    rows = []
    for i in range(4):
        low, high = edges[i], edges[i + 1]
        mask = (fraction >= low) & (fraction <= high if i == 3 else fraction < high)
        if not mask.any():
            continue
        rows.append({
            "quartile": "Q{}".format(i + 1),
            "lesion_fraction_range": [float(low), float(high)],
            "images": int(mask.sum()),
            "mean_per_image_dice": float(dice[mask].mean()),
            "share_of_all_lesion_pixels": float(area[mask].sum() / area.sum()),
        })
    return rows


def main():
    args = parse_args()
    thresholds = np.round(
        np.arange(args.minimum, args.maximum + args.step / 2, args.step), 6
    )

    model = build_model().cuda()
    checkpoint = torch.load(args.checkpoint, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    _, val_loader = make_loader(
        config.val_dataset,
        "Train_Val_text.xlsx",
        args.batch_size,
        text_root=config.task_dataset,
    )
    _, test_loader = make_loader(
        config.test_dataset, "Test_text.xlsx", args.batch_size
    )

    validation = sweep(model, val_loader, thresholds, "validation")
    test = sweep(model, test_loader, thresholds, "test")

    macro_threshold = float(thresholds[int(np.argmax(validation["macro_dice"]))])
    micro_threshold = float(thresholds[int(np.argmax(validation["micro_dice"]))])

    report = {
        "checkpoint": str(args.checkpoint),
        "architecture_version": checkpoint.get("architecture_version"),
        "note": (
            "macro = per-image Dice averaged over images (LViT official "
            "convention, this repo's convention). micro = one global "
            "confusion matrix over all pixels (torchmetrics.Dice default, "
            "LanGuideMedSeg / SGSeg convention). Thresholds are selected on "
            "validation only."
        ),
        "validation": {
            "samples": validation["samples"],
            "macro_selected_threshold": macro_threshold,
            "micro_selected_threshold": micro_threshold,
            "at_0_5": at(validation, thresholds, 0.5),
            "at_macro_selected": at(validation, thresholds, macro_threshold),
            "at_micro_selected": at(validation, thresholds, micro_threshold),
        },
        "test": {
            "samples": test["samples"],
            "at_0_5": at(test, thresholds, 0.5),
            "at_macro_selected": at(test, thresholds, macro_threshold),
            "at_micro_selected": at(test, thresholds, micro_threshold),
            "lesion_size_breakdown_at_0_5": size_breakdown(test),
        },
    }

    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    t = report["test"]["at_macro_selected"]
    m = report["test"]["at_micro_selected"]
    print("\n" + "=" * 66)
    print("TEST, threshold selected on validation")
    print("  macro (our / LViT convention) : Dice {:.6f}  IoU {:.6f}  @ {:.3f}".format(
        t["macro_dice"], t["macro_iou"], t["threshold"]))
    print("  micro (torchmetrics convention): Dice {:.6f}  IoU {:.6f}  @ {:.3f}".format(
        m["micro_dice"], m["micro_iou"], m["threshold"]))
    print("  gap micro - macro             : Dice {:+.6f}  IoU {:+.6f}".format(
        m["micro_dice"] - t["macro_dice"], m["micro_iou"] - t["macro_iou"]))
    print("=" * 66)
    print("Reference points, published numbers:")
    print("  LViT-T        83.66 / 75.11   (macro, verified from official code)")
    print("  LanGuideSeg   89.78 / 81.45   (micro, verified from official code)")
    print("  SGSeg         87.41 / 77.82   (micro, verified from official code)")
    print("  FMISeg        91.21 / 83.84   (convention not verified)")
    print("=" * 66)
    print("Per-image Dice by lesion size (threshold 0.5):")
    for row in report["test"]["lesion_size_breakdown_at_0_5"]:
        print("  {}  lesion {:.1%}-{:.1%}  n={:4d}  Dice {:.4f}  holds {:.1%} of lesion pixels".format(
            row["quartile"],
            row["lesion_fraction_range"][0],
            row["lesion_fraction_range"][1],
            row["images"],
            row["mean_per_image_dice"],
            row["share_of_all_lesion_pixels"],
        ))
    print("\nWritten to", args.output)


if __name__ == "__main__":
    main()

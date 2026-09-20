"""Validation-only BCDH head and image-frequency diagnostics.

This tool never opens the Test split.  Ground-truth masks are used only for
ordinary validation metrics and post-hoc stratification, never to construct a
boundary target or a training loss.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["BETTERLVIT_EXPERIMENT"] = "p6_bcdh_r_v1"

import Config as config
from Load_Dataset import ImageToImage2D, ValGenerator
from nets.BetterLViT import BetterLViT
from utils import read_text


P6_COMMIT = "7217660e6ef16e2a495bab4f20c73403468f55e1"
HEAD_NAMES = ("base", "coarse", "final")
THRESHOLD_HEADS = ("base", "final")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--control-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def git_commit():
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()


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
    return dataset, DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )


def segmentation_metrics(probability, label, threshold=0.5):
    prediction = probability > threshold
    intersection = (prediction & label).sum((1, 2), dtype=torch.float64)
    prediction_sum = prediction.sum((1, 2), dtype=torch.float64)
    label_sum = label.sum((1, 2), dtype=torch.float64)
    denominator = prediction_sum + label_sum
    union = denominator - intersection
    dice = torch.where(
        denominator > 0,
        2.0 * intersection / denominator,
        torch.zeros_like(denominator),
    )
    iou = torch.where(
        union > 0,
        intersection / union,
        torch.zeros_like(union),
    )
    precision = torch.where(
        prediction_sum > 0,
        intersection / prediction_sum,
        torch.zeros_like(intersection),
    )
    recall = torch.where(
        label_sum > 0,
        intersection / label_sum,
        torch.zeros_like(intersection),
    )
    brier = (probability - label.float()).square().mean(
        (1, 2), dtype=torch.float64
    )
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "brier": brier,
        "prediction_pixels": prediction_sum,
    }


def image_frequency_scores(images):
    gray = images.float().mean(dim=1, keepdim=True)
    laplacian_kernel = gray.new_tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    ).view(1, 1, 3, 3)
    laplacian = F.conv2d(
        F.pad(gray, (1, 1, 1, 1), mode="reflect"),
        laplacian_kernel,
    )
    laplacian_energy = laplacian.square().mean((1, 2, 3))

    filters = gray.new_tensor([
        [[1.0, 1.0], [1.0, 1.0]],
        [[-1.0, -1.0], [1.0, 1.0]],
        [[-1.0, 1.0], [-1.0, 1.0]],
        [[1.0, -1.0], [-1.0, 1.0]],
    ]).unsqueeze(1) * 0.5
    coefficients = F.conv2d(gray, filters, stride=2)
    energies = coefficients.square().mean((2, 3))
    haar_ratio = energies[:, 1:].sum(1) / energies.sum(1).clamp_min(1e-12)

    local_mean = F.avg_pool2d(gray, kernel_size=5, stride=1, padding=2)
    normalized_detail = (gray - local_mean).abs().mean((1, 2, 3)) / (
        gray.std((1, 2, 3), unbiased=False).clamp_min(1e-6)
    )
    return laplacian_energy, haar_ratio, normalized_detail


def new_threshold_accumulator(thresholds):
    return {
        head: {
            metric: np.zeros(len(thresholds), dtype=np.float64)
            for metric in ("dice", "precision", "recall")
        }
        for head in THRESHOLD_HEADS
    }


def update_threshold_accumulator(accumulator, probabilities, labels, thresholds):
    for head in THRESHOLD_HEADS:
        probability = probabilities[head]
        for index, threshold in enumerate(thresholds):
            metrics = segmentation_metrics(probability, labels, threshold)
            for metric in accumulator[head]:
                accumulator[head][metric][index] += float(
                    metrics[metric].sum().item()
                )


def summarize_thresholds(accumulator, thresholds, samples):
    result = {}
    for head, metrics in accumulator.items():
        means = {
            name: values / samples
            for name, values in metrics.items()
        }
        best_index = int(np.argmax(means["dice"]))
        result[head] = {
            "best_threshold": float(thresholds[best_index]),
            "best_macro_dice": float(means["dice"][best_index]),
            "precision_at_best": float(means["precision"][best_index]),
            "recall_at_best": float(means["recall"][best_index]),
            "curve": [
                {
                    "threshold": float(threshold),
                    "macro_dice": float(means["dice"][index]),
                    "macro_precision": float(means["precision"][index]),
                    "macro_recall": float(means["recall"][index]),
                }
                for index, threshold in enumerate(thresholds)
            ],
        }
    return result


def summarize_records(records, indices, control_records):
    summary = {"samples": len(indices), "heads": {}}
    for head in HEAD_NAMES:
        summary["heads"][head] = {}
        for metric in ("dice", "iou", "precision", "recall", "brier"):
            values = np.asarray([
                records[index][head][metric] for index in indices
            ])
            summary["heads"][head][metric] = float(values.mean())
    control = {}
    for metric in ("dice", "iou", "precision", "recall", "brier"):
        values = np.asarray([
            control_records[records[index]["name"]][metric]
            for index in indices
        ])
        control[metric] = float(values.mean())
    summary["control"] = control
    summary["delta_vs_control"] = {
        head: {
            metric: summary["heads"][head][metric] - control[metric]
            for metric in control
        }
        for head in HEAD_NAMES
    }
    summary["final_minus_base"] = {
        metric: (
            summary["heads"]["final"][metric]
            - summary["heads"]["base"][metric]
        )
        for metric in control
    }
    return summary


def frequency_quartiles(records, score_name, control_records):
    scores = np.asarray([record[score_name] for record in records])
    order = np.argsort(scores, kind="stable")
    quartiles = []
    for number, indices in enumerate(np.array_split(order, 4), 1):
        entry = summarize_records(records, indices.tolist(), control_records)
        entry.update({
            "quartile": number,
            "score": score_name,
            "score_min": float(scores[indices].min()),
            "score_max": float(scores[indices].max()),
            "score_mean": float(scores[indices].mean()),
        })
        quartiles.append(entry)
    return quartiles


def residual_summary(records):
    keys = (
        "delta_mean",
        "delta_abs_mean",
        "delta_positive_fraction",
        "uncertainty_top20_delta_abs_mean",
        "uncertainty_rest_delta_abs_mean",
    )
    result = {
        key: float(np.mean([record[key] for record in records]))
        for key in keys
    }
    result["top20_minus_rest"] = (
        result["uncertainty_top20_delta_abs_mean"]
        - result["uncertainty_rest_delta_abs_mean"]
    )
    return result


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if config.text_use_lora or config.boundary_loss_weight != 0.0:
        raise RuntimeError("Diagnostic requires frozen text and zero boundary loss")

    checkpoint = torch.load(
        args.checkpoint.resolve(), map_location="cpu", weights_only=True
    )
    if checkpoint.get("experiment_name") != "p6_bcdh_r_v1":
        raise RuntimeError("Expected a P6 checkpoint")
    if checkpoint.get("source_git_commit") != P6_COMMIT:
        raise RuntimeError("P6 checkpoint commit mismatch")
    if not checkpoint.get("bcdh_enabled"):
        raise RuntimeError("P6 checkpoint has BCDH disabled")

    control = json.loads(args.control_json.read_text(encoding="utf-8"))
    if (
        control.get("split") != "validation"
        or control.get("test_split_accessed") is not False
        or control.get("samples") != 1429
    ):
        raise RuntimeError("Invalid C1 validation control JSON")
    control_records = {record["name"]: record for record in control["records"]}

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
    if len(dataset) != len(control_records):
        raise RuntimeError("Control and P6 validation sample counts differ")

    thresholds = np.round(np.arange(0.30, 0.701, 0.01), 2)
    threshold_accumulator = new_threshold_accumulator(thresholds)
    records = []
    with torch.inference_mode():
        for batch, names in tqdm(loader, desc="BCDH frequency diagnostic"):
            images = batch["image"].cuda(non_blocking=True)
            labels = batch["label"].bool().cuda(non_blocking=True)
            outputs = model(
                images,
                batch["input_ids"].cuda(non_blocking=True),
                batch["attention_mask"].cuda(non_blocking=True),
                return_aux=True,
            )
            probabilities = {
                head: outputs[head][:, 0].float()
                for head in HEAD_NAMES
            }
            metrics = {
                head: segmentation_metrics(probabilities[head], labels)
                for head in HEAD_NAMES
            }
            update_threshold_accumulator(
                threshold_accumulator, probabilities, labels, thresholds
            )
            laplacian, haar, normalized_detail = image_frequency_scores(images)
            delta = outputs["delta"][:, 0].float()
            uncertainty = 4.0 * probabilities["coarse"] * (
                1.0 - probabilities["coarse"]
            )
            flattened_uncertainty = uncertainty.flatten(1)
            flattened_delta = delta.flatten(1)
            top_count = max(1, flattened_delta.shape[1] // 5)
            top_indices = torch.topk(
                flattened_uncertainty,
                k=top_count,
                dim=1,
                sorted=False,
            ).indices
            top_abs = flattened_delta.abs().gather(1, top_indices).mean(1)
            all_abs_sum = flattened_delta.abs().sum(1)
            rest_abs = (
                all_abs_sum
                - flattened_delta.abs().gather(1, top_indices).sum(1)
            ) / (flattened_delta.shape[1] - top_count)

            for index, name in enumerate(names):
                name = str(name)
                if name not in control_records:
                    raise RuntimeError("Missing C1 record for {}".format(name))
                record = {
                    "name": name,
                    "label_pixels": int(labels[index].sum().item()),
                    "hf_laplacian_energy": float(laplacian[index].item()),
                    "hf_haar_ratio": float(haar[index].item()),
                    "hf_normalized_local_detail": float(
                        normalized_detail[index].item()
                    ),
                    "delta_mean": float(delta[index].mean().item()),
                    "delta_abs_mean": float(delta[index].abs().mean().item()),
                    "delta_positive_fraction": float(
                        (delta[index] > 0).float().mean().item()
                    ),
                    "uncertainty_top20_delta_abs_mean": float(
                        top_abs[index].item()
                    ),
                    "uncertainty_rest_delta_abs_mean": float(
                        rest_abs[index].item()
                    ),
                }
                for head in HEAD_NAMES:
                    record[head] = {
                        metric: float(values[index].item())
                        for metric, values in metrics[head].items()
                    }
                records.append(record)

    if len(records) != len(dataset):
        raise RuntimeError("Incomplete validation diagnostic")
    indices = list(range(len(records)))
    result = {
        "split": "validation",
        "test_split_accessed": False,
        "samples": len(records),
        "analysis_git_commit": git_commit(),
        "checkpoint_git_commit": checkpoint.get("source_git_commit"),
        "checkpoint_best_epoch": int(checkpoint.get("best_epoch", -1)),
        "text_use_lora": False,
        "boundary_loss_weight": 0.0,
        "training_performed": False,
        "overall": summarize_records(records, indices, control_records),
        "threshold_sweep": summarize_thresholds(
            threshold_accumulator, thresholds, len(records)
        ),
        "residual": residual_summary(records),
        "frequency_quartiles": {
            score: frequency_quartiles(records, score, control_records)
            for score in (
                "hf_laplacian_energy",
                "hf_haar_ratio",
                "hf_normalized_local_detail",
            )
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps({
        key: value for key, value in result.items() if key != "records"
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

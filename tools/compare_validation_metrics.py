"""Paired validation-only comparison for a mechanism-screening pilot."""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1219)
    return parser.parse_args()


def load_validation(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") != "validation":
        raise RuntimeError("Input is not a validation export: {}".format(path))
    if payload.get("test_split_accessed") is not False:
        raise RuntimeError("Input does not prove Test remained untouched.")
    records = {record["name"]: record for record in payload["records"]}
    if len(records) != payload.get("samples"):
        raise RuntimeError("Duplicate or missing validation sample names.")
    return payload, records


def bootstrap_interval(delta, sample_count, seed):
    rng = np.random.default_rng(seed)
    means = np.empty(sample_count, dtype=np.float64)
    for index in range(sample_count):
        selection = rng.integers(0, len(delta), size=len(delta))
        means[index] = delta[selection].mean()
    lower, upper = np.quantile(means, [0.025, 0.975])
    return [float(lower), float(upper)]


def metric_summary(control, candidate, metric, bootstrap_samples, seed):
    delta = candidate - control
    return {
        "control_mean": float(control.mean()),
        "candidate_mean": float(candidate.mean()),
        "mean_delta": float(delta.mean()),
        "bootstrap_95_ci": bootstrap_interval(
            delta,
            bootstrap_samples,
            seed,
        ),
        "win_fraction": float((delta > 0).mean()),
        "loss_fraction": float((delta < 0).mean()),
        "tie_fraction": float((delta == 0).mean()),
        "metric": metric,
    }


def main():
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive.")
    control_payload, control_records = load_validation(args.control)
    candidate_payload, candidate_records = load_validation(args.candidate)
    if set(control_records) != set(candidate_records):
        raise RuntimeError("Control and candidate sample names differ.")
    names = sorted(control_records)
    label_pixels = np.asarray([
        control_records[name]["label_pixels"] for name in names
    ])
    candidate_label_pixels = np.asarray([
        candidate_records[name]["label_pixels"] for name in names
    ])
    if not np.array_equal(label_pixels, candidate_label_pixels):
        raise RuntimeError("Paired validation labels do not match.")

    metrics = {}
    metric_arrays = {}
    higher_is_better = (
        "dice",
        "iou",
        "precision",
        "recall",
        "boundary_f1_tolerance_2",
    )
    for metric_index, metric in enumerate(higher_is_better):
        control = np.asarray([
            control_records[name][metric] for name in names
        ])
        candidate = np.asarray([
            candidate_records[name][metric] for name in names
        ])
        metric_arrays[metric] = (control, candidate)
        metrics[metric] = metric_summary(
            control,
            candidate,
            metric,
            args.bootstrap_samples,
            args.seed + metric_index,
        )

    ordered_indices = np.argsort(label_pixels, kind="stable")
    quartiles = []
    for quartile_index, indices in enumerate(
        np.array_split(ordered_indices, 4),
        1,
    ):
        entry = {
            "quartile": quartile_index,
            "samples": int(len(indices)),
            "label_pixels_min": int(label_pixels[indices].min()),
            "label_pixels_max": int(label_pixels[indices].max()),
        }
        for metric_index, metric in enumerate(higher_is_better):
            control, candidate = metric_arrays[metric]
            entry[metric] = metric_summary(
                control[indices],
                candidate[indices],
                metric,
                args.bootstrap_samples,
                args.seed + 100 * quartile_index + metric_index,
            )
        quartiles.append(entry)

    dice_delta = metrics["dice"]["mean_delta"]
    smallest_delta = quartiles[0]["dice"]["mean_delta"]
    smallest_precision_delta = quartiles[0]["precision"]["mean_delta"]
    overall_precision_delta = metrics["precision"]["mean_delta"]
    boundary_f1_delta = metrics[
        "boundary_f1_tolerance_2"
    ]["mean_delta"]
    control_brier = np.asarray([
        control_records[name]["brier"] for name in names
    ])
    candidate_brier = np.asarray([
        candidate_records[name]["brier"] for name in names
    ])
    brier = metric_summary(
        control_brier,
        candidate_brier,
        "brier_lower_is_better",
        args.bootstrap_samples,
        args.seed + 50,
    )
    metrics["brier_lower_is_better"] = brier
    result = {
        "split": "validation",
        "test_split_accessed": False,
        "control": {
            "experiment": control_payload["experiment"],
            "checkpoint_git_commit": control_payload[
                "checkpoint_git_commit"
            ],
            "best_epoch": control_payload["checkpoint_best_epoch"],
        },
        "candidate": {
            "experiment": candidate_payload["experiment"],
            "checkpoint_git_commit": candidate_payload[
                "checkpoint_git_commit"
            ],
            "best_epoch": candidate_payload["checkpoint_best_epoch"],
        },
        "threshold": candidate_payload["threshold"],
        "samples": len(names),
        "overall": metrics,
        "lesion_size_quartiles": quartiles,
        "pilot_thresholds": {
            "minimum_macro_dice_delta": 0.002,
            "minimum_smallest_quartile_dice_delta": 0.0,
            "minimum_smallest_quartile_precision_delta": 0.0,
            "minimum_macro_precision_delta": 0.0,
            "minimum_boundary_f1_delta": 0.0,
            "maximum_brier_delta": 0.0,
        },
        "passes_numeric_screen": bool(
            dice_delta >= 0.002
            and smallest_delta >= 0.0
            and smallest_precision_delta >= 0.0
            and overall_precision_delta >= 0.0
            and boundary_f1_delta > 0.0
            and brier["mean_delta"] <= 0.0
        ),
        "note": (
            "Numeric screen only. BCDH residual distribution and the "
            "train-validation gap must also be reviewed before extension."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

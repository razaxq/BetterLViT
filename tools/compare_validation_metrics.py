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


def stratified_quartiles(
    scores,
    score_name,
    metric_arrays,
    metric_names,
    bootstrap_samples,
    seed,
):
    ordered_indices = np.argsort(scores, kind="stable")
    quartiles = []
    for quartile_index, indices in enumerate(
        np.array_split(ordered_indices, 4), 1
    ):
        entry = {
            "quartile": quartile_index,
            "samples": int(len(indices)),
            "score": score_name,
            "score_min": float(scores[indices].min()),
            "score_max": float(scores[indices].max()),
        }
        for metric_index, metric in enumerate(metric_names):
            control, candidate = metric_arrays[metric]
            entry[metric] = metric_summary(
                control[indices],
                candidate[indices],
                metric,
                bootstrap_samples,
                seed + 100 * quartile_index + metric_index,
            )
        quartiles.append(entry)
    return quartiles


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

    quartiles = stratified_quartiles(
        label_pixels,
        "label_pixels",
        metric_arrays,
        higher_is_better,
        args.bootstrap_samples,
        args.seed,
    )
    for entry in quartiles:
        entry["label_pixels_min"] = int(entry.pop("score_min"))
        entry["label_pixels_max"] = int(entry.pop("score_max"))

    frequency_quartiles = {}
    frequency_fields = (
        "hf_laplacian_energy",
        "hf_normalized_local_detail",
    )
    if all(
        field in control_records[names[0]]
        and field in candidate_records[names[0]]
        for field in frequency_fields
    ):
        for field_index, field in enumerate(frequency_fields):
            control_scores = np.asarray([
                control_records[name][field] for name in names
            ])
            candidate_scores = np.asarray([
                candidate_records[name][field] for name in names
            ])
            if not np.allclose(control_scores, candidate_scores, atol=1e-9):
                raise RuntimeError(
                    "Control/candidate image frequency scores differ."
                )
            frequency_quartiles[field] = stratified_quartiles(
                control_scores,
                field,
                metric_arrays,
                higher_is_better,
                args.bootstrap_samples,
                args.seed + 1000 * (field_index + 1),
            )

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
    high_frequency_pass = True
    high_frequency_gate = {}
    if frequency_quartiles:
        for field, entries in frequency_quartiles.items():
            highest = entries[-1]
            dice_value = highest["dice"]["mean_delta"]
            precision_value = highest["precision"]["mean_delta"]
            high_frequency_gate[field] = {
                "highest_quartile_dice_delta": dice_value,
                "highest_quartile_precision_delta": precision_value,
                "passes": bool(dice_value >= 0.0 and precision_value >= 0.0),
            }
            high_frequency_pass = (
                high_frequency_pass
                and high_frequency_gate[field]["passes"]
            )
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
        "image_frequency_quartiles": frequency_quartiles,
        "pilot_thresholds": {
            "minimum_macro_dice_delta": 0.002,
            "minimum_smallest_quartile_dice_delta": 0.0,
            "minimum_smallest_quartile_precision_delta": 0.0,
            "minimum_macro_precision_delta": 0.0,
            "minimum_boundary_f1_delta": 0.0,
            "maximum_brier_delta": 0.0,
            "minimum_high_frequency_quartile_dice_delta": 0.0,
            "minimum_high_frequency_quartile_precision_delta": 0.0,
        },
        "high_frequency_gate": high_frequency_gate,
        "passes_numeric_screen": bool(
            dice_delta >= 0.002
            and smallest_delta >= 0.0
            and smallest_precision_delta >= 0.0
            and overall_precision_delta >= 0.0
            and boundary_f1_delta > 0.0
            and brier["mean_delta"] <= 0.0
            and high_frequency_pass
        ),
        "note": (
            "Numeric screen only. Refiner residual distribution and the "
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

"""Create colour-coded comparisons of ground-truth and predicted masks."""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import Config as config


COLOURS_BGR = {
    "overlap": (0, 200, 0),
    "prediction_only": (0, 0, 255),
    "ground_truth_only": (255, 0, 0),
}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Combine test ground-truth and prediction masks into one "
            "colour-coded image per sample."
        )
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        help="Directory containing ground-truth masks.",
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        help=(
            "Directory containing *_pred.png masks. By default the newest "
            "complete test_predictions_threshold_* directory is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory for comparison images.",
    )
    parser.add_argument(
        "--background",
        choices=("black", "original"),
        default="black",
        help="Use a black background or the corresponding test image.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Directory containing original test images.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.65,
        help="Colour opacity when --background original is used.",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Do not add the compact TP/FP/FN legend above each image.",
    )
    return parser.parse_args()


def resolve_repo_path(path):
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def latest_complete_prediction_dir():
    experiment_root = (
        REPO_ROOT / config.task_name / config.model_name
    )
    candidates = [
        path
        for path in experiment_root.glob(
            "*/test_predictions_threshold_*"
        )
        if (
            path.is_dir()
            and not path.name.endswith("_INCOMPLETE")
            and (path / "manifest.json").is_file()
        )
    ]
    if not candidates:
        raise FileNotFoundError(
            "No complete test prediction directory was found."
        )
    return max(
        candidates,
        key=lambda path: (path / "manifest.json").stat().st_mtime,
    ).resolve()


def build_file_index(directory, kind):
    index = {}
    for path in sorted(directory.iterdir()):
        if (
            not path.is_file()
            or path.suffix.lower() not in IMAGE_SUFFIXES
        ):
            continue
        key = path.stem
        if kind == "ground_truth" and key.startswith("mask_"):
            key = key[len("mask_"):]
        if kind == "prediction":
            if not key.endswith("_pred"):
                continue
            key = key[:-len("_pred")]
        if key in index:
            raise ValueError(
                f"Duplicate {kind} key {key!r}: "
                f"{index[key]} and {path}"
            )
        index[key] = path
    return index


def find_original_image(image_dir, key):
    matches = [
        path
        for path in image_dir.glob(f"{key}.*")
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
        )
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one original image for {key!r}, found {len(matches)}."
        )
    return matches[0]


def read_binary_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise OSError(f"Unable to read mask: {path}")
    return mask > 0


def mask_metrics(ground_truth, prediction):
    overlap = ground_truth & prediction
    prediction_only = prediction & ~ground_truth
    ground_truth_only = ground_truth & ~prediction
    intersection = int(overlap.sum())
    prediction_sum = int(prediction.sum())
    ground_truth_sum = int(ground_truth.sum())
    denominator = prediction_sum + ground_truth_sum
    union = denominator - intersection
    dice = (
        2.0 * intersection / denominator
        if denominator
        else 0.0
    )
    iou = intersection / union if union else 0.0
    return {
        "dice": dice,
        "iou": iou,
        "overlap_pixels": intersection,
        "prediction_only_pixels": int(prediction_only.sum()),
        "ground_truth_only_pixels": int(ground_truth_only.sum()),
    }


def colourise_masks(ground_truth, prediction, background, alpha):
    overlap = ground_truth & prediction
    prediction_only = prediction & ~ground_truth
    ground_truth_only = ground_truth & ~prediction

    if background is None:
        comparison = np.zeros(
            (*ground_truth.shape, 3),
            dtype=np.uint8,
        )
    else:
        comparison = background.copy()

    for mask, colour_name in (
        (overlap, "overlap"),
        (prediction_only, "prediction_only"),
        (ground_truth_only, "ground_truth_only"),
    ):
        colour = np.asarray(
            COLOURS_BGR[colour_name],
            dtype=np.float32,
        )
        if background is None:
            comparison[mask] = colour.astype(np.uint8)
        else:
            comparison[mask] = np.clip(
                (1.0 - alpha) * comparison[mask].astype(np.float32)
                + alpha * colour,
                0,
                255,
            ).astype(np.uint8)
    return comparison


def add_compact_legend(image):
    legend_height = 28
    canvas = np.zeros(
        (image.shape[0] + legend_height, image.shape[1], 3),
        dtype=np.uint8,
    )
    canvas[legend_height:] = image
    items = (
        ("TP", COLOURS_BGR["overlap"]),
        ("FP", COLOURS_BGR["prediction_only"]),
        ("FN", COLOURS_BGR["ground_truth_only"]),
    )
    item_width = image.shape[1] // len(items)
    for index, (label, colour) in enumerate(items):
        x = index * item_width + 7
        cv2.rectangle(
            canvas,
            (x, 8),
            (x + 12, 20),
            colour,
            thickness=-1,
        )
        cv2.putText(
            canvas,
            label,
            (x + 17, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return canvas


def write_legend(output_dir):
    legend = np.zeros((110, 640, 3), dtype=np.uint8)
    entries = (
        ("TP / overlap", COLOURS_BGR["overlap"]),
        ("FP / prediction only", COLOURS_BGR["prediction_only"]),
        ("FN / ground truth only", COLOURS_BGR["ground_truth_only"]),
    )
    for index, (label, colour) in enumerate(entries):
        y = 12 + index * 32
        cv2.rectangle(
            legend,
            (15, y),
            (37, y + 22),
            colour,
            thickness=-1,
        )
        cv2.putText(
            legend,
            label,
            (50, y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    legend_path = output_dir / "legend.png"
    if not cv2.imwrite(str(legend_path), legend):
        raise OSError(f"Unable to save legend: {legend_path}")


def main():
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1.")

    ground_truth_dir = (
        args.ground_truth_dir.resolve()
        if args.ground_truth_dir
        else resolve_repo_path(config.test_dataset) / "labelcol"
    )
    prediction_dir = (
        args.prediction_dir.resolve()
        if args.prediction_dir
        else latest_complete_prediction_dir()
    )
    image_dir = (
        args.image_dir.resolve()
        if args.image_dir
        else resolve_repo_path(config.test_dataset) / "img"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else prediction_dir.parent
        / prediction_dir.name.replace(
            "test_predictions_",
            "test_comparisons_",
            1,
        )
    )

    if not ground_truth_dir.is_dir():
        raise FileNotFoundError(
            f"Ground-truth directory not found: {ground_truth_dir}"
        )
    if not prediction_dir.is_dir():
        raise FileNotFoundError(
            f"Prediction directory not found: {prediction_dir}"
        )
    if args.background == "original" and not image_dir.is_dir():
        raise FileNotFoundError(
            f"Original-image directory not found: {image_dir}"
        )

    ground_truth_files = build_file_index(
        ground_truth_dir,
        "ground_truth",
    )
    prediction_files = build_file_index(
        prediction_dir,
        "prediction",
    )
    if not ground_truth_files or not prediction_files:
        raise RuntimeError(
            "Ground-truth and prediction directories must contain masks."
        )
    missing_predictions = sorted(
        set(ground_truth_files) - set(prediction_files)
    )
    missing_ground_truth = sorted(
        set(prediction_files) - set(ground_truth_files)
    )
    if missing_predictions or missing_ground_truth:
        raise RuntimeError(
            "Mask pairing failed. "
            f"Missing predictions: {len(missing_predictions)}; "
            f"missing ground truth: {len(missing_ground_truth)}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for key in tqdm(
        sorted(ground_truth_files),
        desc="Compare masks",
        unit="image",
        ncols=80,
    ):
        ground_truth = read_binary_mask(ground_truth_files[key])
        prediction = read_binary_mask(prediction_files[key])
        if prediction.shape != ground_truth.shape:
            prediction = cv2.resize(
                prediction.astype(np.uint8),
                (ground_truth.shape[1], ground_truth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        background = None
        if args.background == "original":
            original_path = find_original_image(image_dir, key)
            original = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
            if original is None:
                raise OSError(
                    f"Unable to read original image: {original_path}"
                )
            original = cv2.resize(
                original,
                (ground_truth.shape[1], ground_truth.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
            background = cv2.cvtColor(
                original,
                cv2.COLOR_GRAY2BGR,
            )

        comparison = colourise_masks(
            ground_truth,
            prediction,
            background,
            args.alpha,
        )
        if not args.no_legend:
            comparison = add_compact_legend(comparison)
        comparison_path = output_dir / f"{key}_comparison.png"
        if not cv2.imwrite(str(comparison_path), comparison):
            raise OSError(
                f"Unable to save comparison: {comparison_path}"
            )

        record = {"image": key}
        record.update(mask_metrics(ground_truth, prediction))
        records.append(record)

    metrics_path = output_dir / "comparison_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=records[0].keys(),
        )
        writer.writeheader()
        writer.writerows(records)

    write_legend(output_dir)
    summary = {
        "ground_truth_directory": str(ground_truth_dir),
        "prediction_directory": str(prediction_dir),
        "output_directory": str(output_dir),
        "background": args.background,
        "samples": len(records),
        "mean_dice": float(
            np.mean([record["dice"] for record in records])
        ),
        "mean_iou": float(
            np.mean([record["iou"] for record in records])
        ),
        "colours_rgb": {
            "overlap_tp": [0, 200, 0],
            "prediction_only_fp": [255, 0, 0],
            "ground_truth_only_fn": [0, 0, 255],
            "background": [0, 0, 0],
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

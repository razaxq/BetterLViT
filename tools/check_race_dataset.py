"""Exercise one real QaTa train sample through the RACE data transform."""

import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("BETTERLVIT_EXPERIMENT", "p8_race_fuse_v1")

import Config as config  # noqa: E402
from Load_Dataset import ImageToImage2D, RandomGenerator  # noqa: E402
from utils import read_text  # noqa: E402


def main():
    text = read_text(os.path.join(
        config.task_dataset, "Train_Val_text.xlsx"
    ))
    dataset = ImageToImage2D(
        config.train_dataset,
        config.task_name,
        text,
        RandomGenerator([config.img_size, config.img_size]),
        image_size=config.img_size,
    )
    sample, name = dataset[0]
    expected = {
        "image": (3, config.img_size, config.img_size),
        "label": (config.img_size, config.img_size),
        "input_ids": (config.text_max_len,),
        "attention_mask": (config.text_max_len,),
        "race_slot_targets": (9,),
        "race_zone_basis": (6, config.img_size, config.img_size),
    }
    actual = {key: tuple(value.shape) for key, value in sample.items()}
    if actual != expected:
        raise RuntimeError("Unexpected transformed sample shapes: {}".format(actual))
    basis = sample["race_zone_basis"]
    if not torch.isfinite(basis).all() or basis.min() < 0 or basis.max() > 1:
        raise RuntimeError("Invalid transformed RACE zone basis")
    if (basis.sum(dim=0) > 1.0).any():
        raise RuntimeError("Transformed RACE zones overlap")
    print(json.dumps({
        "status": "ok",
        "sample": name,
        "dataset_samples": len(dataset),
        "shapes": {key: list(value) for key, value in actual.items()},
        "zone_basis_pixels": float(basis.sum()),
        "slot_targets": sample["race_slot_targets"].tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()

"""Load and validate immutable training/validation split manifests."""

import json
from pathlib import Path

from reproducibility import canonical_sha256, sha256_file


SCHEMA_VERSION = 1
GROUPED_PROTOCOL = "known_patient_grouped_sensitivity_v1"
EXPECTED_POOL_SAMPLES = 7145
EXPECTED_TRAIN_SAMPLES = 5716
EXPECTED_VALIDATION_SAMPLES = 1429


def _resolve_inside(root, relative_path):
    relative = Path(str(relative_path))
    if relative.is_absolute():
        raise RuntimeError("Manifest paths must be relative: {}".format(relative))
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            "Manifest path escapes dataset root: {}".format(relative)
        ) from exc
    return resolved


def _validate_record(record, dataset_root, split_name):
    required = {
        "image_name",
        "mask_name",
        "text_key",
        "image_path",
        "mask_path",
        "legacy_split",
        "group_kind",
        "group_id",
    }
    if not isinstance(record, dict) or not required.issubset(record):
        raise RuntimeError(
            "Invalid {} manifest record; required keys are {}".format(
                split_name,
                sorted(required),
            )
        )
    image_name = str(record["image_name"])
    mask_name = str(record["mask_name"])
    text_key = str(record["text_key"])
    if mask_name != "mask_" + image_name or text_key != mask_name:
        raise RuntimeError(
            "Manifest image/mask/text pairing mismatch for {}".format(
                image_name
            )
        )
    normalized_paths = (
        str(record["image_path"]).replace("\\", "/"),
        str(record["mask_path"]).replace("\\", "/"),
    )
    if any("Test_Folder" in path.split("/") for path in normalized_paths):
        raise RuntimeError("Grouped manifest must never reference Test_Folder")
    image_path = _resolve_inside(dataset_root, record["image_path"])
    mask_path = _resolve_inside(dataset_root, record["mask_path"])
    if not image_path.is_file() or not mask_path.is_file():
        raise FileNotFoundError(
            "Manifest sample is missing data: {} / {}".format(
                image_path,
                mask_path,
            )
        )
    return {
        "image_name": image_name,
        "mask_name": mask_name,
        "text_key": text_key,
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "legacy_split": str(record["legacy_split"]),
        "group_kind": str(record["group_kind"]),
        "group_id": str(record["group_id"]),
    }


def load_grouped_manifest(path, dataset_root, expected_protocol):
    manifest_path = Path(path).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Split manifest not found: {}".format(manifest_path)
        )
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            "Dataset root not found: {}".format(dataset_root)
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Unsupported split manifest schema")
    if manifest.get("protocol") != expected_protocol:
        raise RuntimeError(
            "Split protocol mismatch: expected {!r}, found {!r}".format(
                expected_protocol,
                manifest.get("protocol"),
            )
        )
    if expected_protocol != GROUPED_PROTOCOL:
        raise RuntimeError("This loader only accepts the grouped protocol")
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise RuntimeError("Split manifest has no splits object")
    raw_train = splits.get("train")
    raw_validation = splits.get("validation")
    if not isinstance(raw_train, list) or not isinstance(raw_validation, list):
        raise RuntimeError("Split manifest train/validation must be arrays")
    if len(raw_train) != EXPECTED_TRAIN_SAMPLES:
        raise RuntimeError("Grouped train count must be 5716")
    if len(raw_validation) != EXPECTED_VALIDATION_SAMPLES:
        raise RuntimeError("Grouped validation count must be 1429")

    # The builder hashes the canonical split payload. Recompute it before any
    # filesystem resolution so the same portable manifest verifies everywhere.
    expected_content_hash = manifest.get("split_content_sha256")
    actual_content_hash = canonical_sha256(splits)
    if expected_content_hash != actual_content_hash:
        raise RuntimeError("Split manifest canonical content hash mismatch")

    train = [
        _validate_record(record, dataset_root, "train")
        for record in raw_train
    ]
    validation = [
        _validate_record(record, dataset_root, "validation")
        for record in raw_validation
    ]
    all_records = train + validation
    image_names = [record["image_name"] for record in all_records]
    if len(all_records) != EXPECTED_POOL_SAMPLES:
        raise RuntimeError("Grouped manifest pool count must be 7145")
    if len(set(image_names)) != len(image_names):
        raise RuntimeError("Grouped manifest has duplicate image names")

    train_known = {
        record["group_id"]
        for record in train
        if record["group_kind"] == "known_subject"
    }
    validation_known = {
        record["group_id"]
        for record in validation
        if record["group_kind"] == "known_subject"
    }
    overlap = train_known & validation_known
    if overlap:
        raise RuntimeError(
            "Grouped manifest leaks {} known subjects".format(len(overlap))
        )
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "train_records": train,
        "validation_records": validation,
        "known_subject_overlap": 0,
    }

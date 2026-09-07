"""Build the deterministic known-patient-grouped validation sensitivity split.

Only the original Covid19 Train_Folder and Val_Folder are read.  The official
test folder is outside this script's input surface by design.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re

import pandas as pd


SCHEMA_VERSION = 1
PROTOCOL = "known_patient_grouped_sensitivity_v1"
ALGORITHM_VERSION = "sha256_stratified_exact_subset_v1"
DEFAULT_SEED = 1219
EXPECTED_TRAIN = 5716
EXPECTED_VALIDATION = 1429
EXPECTED_KNOWN_POOL = 4194
EXPECTED_ANONYMOUS_POOL = 2951
TARGET_KNOWN_VALIDATION = 839
TARGET_ANONYMOUS_VALIDATION = 590
SUBJECT_PATTERN = re.compile(r"^sub-(S\d+)_")
ANONYMOUS_PATTERN = re.compile(r"^covid_\d+\.png$")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/Covid19"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "protocols/splits/known_patient_grouped_seed1219.json"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def canonical_sha256(value):
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_group_key(group_id, seed):
    payload = "{}\0{}".format(int(seed), group_id).encode("utf-8")
    return hashlib.sha256(payload).digest(), group_id.encode("utf-8")


def read_text_keys(workbook_path):
    frame = pd.read_excel(workbook_path)
    required = {"Image", "Description"}
    if not required.issubset(frame.columns):
        raise RuntimeError(
            "Text workbook must contain Image and Description columns"
        )
    if frame["Image"].isna().any() or frame["Description"].isna().any():
        raise RuntimeError("Text workbook contains blank keys/descriptions")
    keys = [str(value) for value in frame["Image"].tolist()]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Text workbook contains duplicate Image keys")
    return set(keys)


def scan_legacy_split(dataset_root, folder_name, legacy_split, text_keys):
    folder = dataset_root / folder_name
    image_root = folder / "img"
    mask_root = folder / "labelcol"
    if not image_root.is_dir() or not mask_root.is_dir():
        raise FileNotFoundError("Missing legacy split: {}".format(folder))
    image_names = sorted(
        path.name for path in image_root.iterdir() if path.is_file()
    )
    mask_names = sorted(
        path.name for path in mask_root.iterdir() if path.is_file()
    )
    expected_masks = {"mask_" + name for name in image_names}
    if expected_masks != set(mask_names):
        raise RuntimeError(
            "{} image/mask pairing is incomplete".format(folder_name)
        )
    records = []
    for image_name in image_names:
        mask_name = "mask_" + image_name
        if mask_name not in text_keys:
            raise RuntimeError(
                "Text workbook lacks {}".format(mask_name)
            )
        subject_match = SUBJECT_PATTERN.match(image_name)
        if subject_match:
            group_kind = "known_subject"
            group_id = "subject:" + subject_match.group(1)
        elif ANONYMOUS_PATTERN.match(image_name):
            group_kind = "anonymous_singleton"
            group_id = "anonymous:" + Path(image_name).stem
        else:
            raise RuntimeError(
                "Unrecognized Covid19 sample name: {}".format(image_name)
            )
        records.append({
            "image_name": image_name,
            "mask_name": mask_name,
            "text_key": mask_name,
            "image_path": (
                Path(folder_name) / "img" / image_name
            ).as_posix(),
            "mask_path": (
                Path(folder_name) / "labelcol" / mask_name
            ).as_posix(),
            "legacy_split": legacy_split,
            "group_kind": group_kind,
            "group_id": group_id,
        })
    return records


def choose_exact_group_subset(groups, target_images, seed):
    """Choose whole groups with an exact image total using stable subset DP."""
    ordered = sorted(groups, key=lambda group: stable_group_key(group, seed))
    predecessor = [None] * (target_images + 1)
    predecessor[0] = (-1, None)
    for group_id in ordered:
        size = len(groups[group_id])
        if size > target_images:
            continue
        for total in range(target_images, size - 1, -1):
            if predecessor[total] is None and predecessor[total - size] is not None:
                predecessor[total] = (total - size, group_id)
        if predecessor[target_images] is not None:
            break
    if predecessor[target_images] is None:
        raise RuntimeError(
            "Cannot select exactly {} grouped images".format(target_images)
        )
    selected = set()
    total = target_images
    while total:
        previous, group_id = predecessor[total]
        selected.add(group_id)
        total = previous
    if sum(len(groups[group_id]) for group_id in selected) != target_images:
        raise RuntimeError("Exact grouped subset reconstruction failed")
    return selected


def build_manifest(dataset_root, seed):
    dataset_root = dataset_root.resolve()
    workbook = dataset_root / "Train_Folder" / "Train_Val_text.xlsx"
    if not workbook.is_file():
        raise FileNotFoundError(workbook)
    text_keys = read_text_keys(workbook)
    pool = (
        scan_legacy_split(
            dataset_root,
            "Train_Folder",
            "train",
            text_keys,
        )
        + scan_legacy_split(
            dataset_root,
            "Val_Folder",
            "validation",
            text_keys,
        )
    )
    names = [record["image_name"] for record in pool]
    if len(pool) != EXPECTED_TRAIN + EXPECTED_VALIDATION:
        raise RuntimeError("Expected exactly 7,145 train+validation samples")
    if len(names) != len(set(names)):
        raise RuntimeError("Legacy train+validation pool has duplicate names")
    if {record["text_key"] for record in pool} != text_keys:
        raise RuntimeError(
            "Text workbook keys do not exactly equal the 7,145 masks"
        )

    known_groups = {}
    anonymous_groups = {}
    for record in pool:
        destination = (
            known_groups
            if record["group_kind"] == "known_subject"
            else anonymous_groups
        )
        destination.setdefault(record["group_id"], []).append(record)
    known_count = sum(len(records) for records in known_groups.values())
    anonymous_count = sum(
        len(records) for records in anonymous_groups.values()
    )
    if known_count != EXPECTED_KNOWN_POOL:
        raise RuntimeError("Expected 4,194 known-subject images")
    if anonymous_count != EXPECTED_ANONYMOUS_POOL:
        raise RuntimeError("Expected 2,951 anonymous images")
    if any(len(records) != 1 for records in anonymous_groups.values()):
        raise RuntimeError("Anonymous pseudo-groups must be singletons")

    validation_known_groups = choose_exact_group_subset(
        known_groups,
        TARGET_KNOWN_VALIDATION,
        seed,
    )
    ordered_anonymous = sorted(
        anonymous_groups,
        key=lambda group: stable_group_key(group, seed),
    )
    validation_anonymous_groups = set(
        ordered_anonymous[:TARGET_ANONYMOUS_VALIDATION]
    )
    validation_groups = (
        validation_known_groups | validation_anonymous_groups
    )
    validation = sorted(
        [record for record in pool if record["group_id"] in validation_groups],
        key=lambda record: record["image_name"],
    )
    train = sorted(
        [record for record in pool if record["group_id"] not in validation_groups],
        key=lambda record: record["image_name"],
    )
    if len(train) != EXPECTED_TRAIN or len(validation) != EXPECTED_VALIDATION:
        raise RuntimeError("Grouped split counts are not 5,716/1,429")
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
    if train_known & validation_known:
        raise RuntimeError("Known subject appears in both grouped splits")

    splits = {"train": train, "validation": validation}
    pool_membership = [
        {
            key: record[key]
            for key in (
                "image_name",
                "mask_name",
                "legacy_split",
                "group_kind",
                "group_id",
            )
        }
        for record in sorted(pool, key=lambda record: record["image_name"])
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "algorithm": {
            "version": ALGORITHM_VERSION,
            "seed": int(seed),
            "known_group_rule": "^sub-(S\\d+)_ grouped by subject",
            "anonymous_group_rule": (
                "covid_N treated as singleton pseudo-group because subject "
                "metadata is unavailable"
            ),
            "source_stratification": {
                "known_validation_images": TARGET_KNOWN_VALIDATION,
                "anonymous_validation_images": TARGET_ANONYMOUS_VALIDATION,
            },
        },
        "source": {
            "folders": ["Train_Folder", "Val_Folder"],
            "pool_samples": len(pool),
            "pool_membership_sha256": canonical_sha256(pool_membership),
            "text_workbook": "Train_Folder/Train_Val_text.xlsx",
            "text_keys": len(text_keys),
        },
        "counts": {
            "train": len(train),
            "validation": len(validation),
            "known_pool_images": known_count,
            "known_pool_subjects": len(known_groups),
            "known_validation_images": sum(
                record["group_kind"] == "known_subject"
                for record in validation
            ),
            "known_validation_subjects": len(validation_known),
            "anonymous_validation_images": sum(
                record["group_kind"] == "anonymous_singleton"
                for record in validation
            ),
            "known_subject_overlap": 0,
        },
        "limitations": [
            "The 2,951 covid_N samples lack patient identifiers and are "
            "treated as singleton pseudo-groups.",
            "This is a known-patient-grouped sensitivity split, not a claim "
            "of complete patient independence.",
        ],
        "split_content_sha256": canonical_sha256(splits),
        "splits": splits,
    }


def main():
    args = parse_args()
    manifest = build_manifest(args.dataset_root, args.seed)
    serialized = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = output.read_text(encoding="utf-8")
        if existing != serialized:
            raise FileExistsError(
                "Refusing to overwrite a different manifest: {}".format(output)
            )
    else:
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(output)
    print(json.dumps(manifest["counts"], indent=2))
    print("split_content_sha256={}".format(
        manifest["split_content_sha256"]
    ))
    print("saved={}".format(output))


if __name__ == "__main__":
    main()

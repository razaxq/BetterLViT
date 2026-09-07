"""Validate a strict V4-B epoch-boundary checkpoint without touching test data."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--completed-epochs", type=int, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.completed_epochs < 1:
        raise ValueError("--completed-epochs must be positive")
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        raise ValueError("--commit must be a 40-hex hash")
    if not re.fullmatch(r"[0-9a-f]{64}", args.manifest_sha256):
        raise ValueError("--manifest-sha256 must be a SHA-256")
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    completed = args.completed_epochs
    if int(checkpoint.get("epoch", -1)) != completed - 1:
        raise RuntimeError("Checkpoint epoch index is inconsistent")
    if checkpoint.get("best_model") is not False:
        raise RuntimeError("Epoch gate must inspect rolling last_model")
    if checkpoint.get("architecture_version") != "fam_eppa_v4b":
        raise RuntimeError("Checkpoint architecture is not fam_eppa_v4b")
    if checkpoint.get("reproducibility_protocol") != "strict_paired_v1":
        raise RuntimeError("Checkpoint lacks strict reproducibility protocol")
    if checkpoint.get("text_modality_dropout_prob") != 0.0:
        raise RuntimeError("Grouped rebaseline checkpoint must use p=0.0")
    history = checkpoint.get("epoch_history")
    if not isinstance(history, list) or len(history) != completed:
        raise RuntimeError("Checkpoint epoch history length is inconsistent")
    if int(history[-1].get("epoch", -1)) != completed:
        raise RuntimeError("Checkpoint final history row is inconsistent")
    if not isinstance(checkpoint.get("optimizer"), dict):
        raise RuntimeError("Checkpoint lacks optimizer state")
    if not isinstance(checkpoint.get("lr_scheduler"), dict):
        raise RuntimeError("Checkpoint lacks scheduler state")
    if not isinstance(checkpoint.get("rng_state"), dict):
        raise RuntimeError("Checkpoint lacks RNG state")
    fingerprint = checkpoint.get("run_fingerprint")
    if not isinstance(fingerprint, dict):
        raise RuntimeError("Checkpoint lacks run fingerprint")
    if fingerprint.get("git_commit") != args.commit:
        raise RuntimeError("Checkpoint source commit mismatch")
    if fingerprint.get("split_manifest_sha256") != args.manifest_sha256:
        raise RuntimeError("Checkpoint split manifest mismatch")
    if fingerprint.get("split_protocol") != (
        "known_patient_grouped_sensitivity_v1"
    ):
        raise RuntimeError("Checkpoint split protocol mismatch")
    if fingerprint.get("train_samples") != 5716:
        raise RuntimeError("Checkpoint train count mismatch")
    if fingerprint.get("validation_samples") != 1429:
        raise RuntimeError("Checkpoint validation count mismatch")
    for field in ("dataset_artifacts_sha256", "text_workbook_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(fingerprint.get(field, ""))):
            raise RuntimeError("Checkpoint {} is missing".format(field))
    if not isinstance(fingerprint.get("package_versions"), dict):
        raise RuntimeError("Checkpoint package versions are missing")
    if not fingerprint.get("python_version"):
        raise RuntimeError("Checkpoint Python version is missing")
    claimed_fingerprint = fingerprint.get("fingerprint_sha256")
    fingerprint_payload = dict(fingerprint)
    fingerprint_payload.pop("fingerprint_sha256", None)
    canonical = json.dumps(
        fingerprint_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    actual_fingerprint = hashlib.sha256(canonical).hexdigest()
    if claimed_fingerprint != actual_fingerprint:
        raise RuntimeError("Checkpoint run fingerprint is internally invalid")
    print("checkpoint={}".format(checkpoint_path))
    print("completed_epochs={}".format(completed))
    print("fingerprint_sha256={}".format(
        fingerprint.get("fingerprint_sha256")
    ))
    print("Strict V4-B epoch checkpoint gate passed.")


if __name__ == "__main__":
    main()

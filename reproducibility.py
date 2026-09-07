"""Strict paired-experiment reproducibility utilities.

The original training path seeded global RNGs but let model construction,
sampling, workers, and augmentation share state.  This module separates those
streams and records enough state to resume exactly at an epoch boundary.
"""

import hashlib
import importlib.metadata
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


PROTOCOL_VERSION = "strict_paired_v1"
REQUIRED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def require_process_environment(seed):
    """Fail before training if process-start determinism was not requested."""
    expected_seed = str(int(seed))
    actual_hash_seed = os.environ.get("PYTHONHASHSEED")
    if actual_hash_seed != expected_seed:
        raise RuntimeError(
            "PYTHONHASHSEED must be set before Python starts: expected {}, "
            "found {!r}".format(expected_seed, actual_hash_seed)
        )
    actual_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if actual_cublas != REQUIRED_CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be {!r}, found {!r}".format(
                REQUIRED_CUBLAS_WORKSPACE_CONFIG,
                actual_cublas,
            )
        )


def configure_determinism(seed):
    """Configure deterministic CUDA execution and initialize global streams."""
    require_process_environment(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def reset_global_seed(seed):
    """Reset model/training-op state without touching loader generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def make_generator(seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def seed_worker(worker_id):
    """Seed every augmentation RNG from PyTorch's dedicated worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def capture_rng_state(generators):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else []
        ),
        "generators": {
            name: generator.get_state()
            for name, generator in sorted(generators.items())
        },
    }
    return state


def restore_rng_state(state, generators):
    required = {"python", "numpy", "torch_cpu", "torch_cuda", "generators"}
    if not isinstance(state, dict) or set(state) != required:
        raise RuntimeError("Checkpoint RNG state is missing or incompatible")
    expected_names = set(generators)
    checkpoint_names = set(state["generators"])
    if checkpoint_names != expected_names:
        raise RuntimeError(
            "Checkpoint generator names mismatch: expected {}, found {}".format(
                sorted(expected_names),
                sorted(checkpoint_names),
            )
        )
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        cuda_states = state["torch_cuda"]
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "Checkpoint CUDA RNG state count does not match visible GPUs"
            )
        torch.cuda.set_rng_state_all(cuda_states)
    elif state["torch_cuda"]:
        raise RuntimeError("CUDA RNG state cannot be restored without CUDA")
    for name, generator in generators.items():
        generator.set_state(state["generators"][name])


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_sha256(values):
    payload = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(value):
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def named_file_set_sha256(named_paths):
    """Hash immutable artifacts by logical name and file content."""
    digest = hashlib.sha256()
    digest.update(b"betterlvit-named-file-set-v1\0")
    seen = set()
    for logical_name, path in sorted(
        (str(name), Path(path).resolve()) for name, path in named_paths
    ):
        if logical_name in seen:
            raise RuntimeError(
                "Duplicate logical artifact name: {}".format(logical_name)
            )
        seen.add(logical_name)
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(logical_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def package_versions():
    """Record data/model runtime versions that can affect exact replay."""
    distributions = (
        "numpy",
        "scipy",
        "opencv-python",
        "opencv-python-headless",
        "transformers",
        "tokenizers",
        "peft",
        "pandas",
        "openpyxl",
        "Pillow",
        "torchvision",
        "tensorboardX",
        "ml-collections",
    )
    versions = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def build_run_fingerprint(
    *,
    seed,
    split_protocol,
    split_manifest_path,
    split_manifest_sha256,
    train_names,
    validation_names,
    text_modality_dropout_prob,
    text_modality_dropout_prompt,
    training_configuration,
    git_commit,
    dataset_artifacts_sha256,
    text_workbook_sha256,
):
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "seed": int(seed),
        "split_protocol": split_protocol,
        "split_manifest_path": split_manifest_path,
        "split_manifest_sha256": split_manifest_sha256,
        "train_samples": len(train_names),
        "validation_samples": len(validation_names),
        "train_order_sha256": sequence_sha256(train_names),
        "validation_order_sha256": sequence_sha256(validation_names),
        "text_modality_dropout_prob": float(text_modality_dropout_prob),
        "text_modality_dropout_prompt": text_modality_dropout_prompt,
        "dataset_artifacts_sha256": dataset_artifacts_sha256,
        "text_workbook_sha256": text_workbook_sha256,
        "training_configuration": training_configuration,
        "git_commit": git_commit,
        "python_version": sys.version,
        "package_versions": package_versions(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "cuda_compute_capability": (
            list(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available()
            else None
        ),
    }
    payload["fingerprint_sha256"] = canonical_sha256(payload)
    return payload


def validate_resume_fingerprint(checkpoint_fingerprint, expected_fingerprint):
    if not isinstance(checkpoint_fingerprint, dict):
        raise RuntimeError("Checkpoint lacks strict reproducibility metadata")
    checkpoint_hash = checkpoint_fingerprint.get("fingerprint_sha256")
    expected_hash = expected_fingerprint.get("fingerprint_sha256")
    if checkpoint_hash != expected_hash:
        raise RuntimeError(
            "Checkpoint reproducibility fingerprint mismatch: {} != {}".format(
                checkpoint_hash,
                expected_hash,
            )
        )

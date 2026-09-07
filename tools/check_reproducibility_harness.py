"""Exercise deterministic sampling, augmentation, and epoch resume on real data."""

import gc
import hashlib
import os
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import Config as config  # noqa: E402
from Load_Dataset import ImageToImage2D, RandomGenerator  # noqa: E402
from Train_one_epoch import apply_text_modality_dropout  # noqa: E402
from reproducibility import (  # noqa: E402
    configure_determinism,
    make_generator,
    seed_worker,
)
from split_protocol import GROUPED_PROTOCOL, load_grouped_manifest  # noqa: E402
from utils import read_text  # noqa: E402


def build_dataset():
    grouped = load_grouped_manifest(
        config.split_manifest_path,
        config.dataset_root,
        config.split_protocol,
    )
    text = read_text(
        os.path.join(config.task_dataset, "Train_Val_text.xlsx")
    )
    return ImageToImage2D(
        config.train_dataset,
        config.task_name,
        text,
        transforms.Compose(
            [RandomGenerator([config.img_size, config.img_size])]
        ),
        image_size=config.img_size,
        sample_records=grouped["train_records"],
    )


def build_loader(dataset, sampler_generator, worker_generator):
    return DataLoader(
        dataset,
        batch_size=4,
        sampler=RandomSampler(dataset, generator=sampler_generator),
        num_workers=2,
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=worker_generator,
    )


def collect_prefix(loader, batches=2):
    digest = hashlib.sha256()
    iterator = iter(loader)
    try:
        for _ in range(batches):
            batch, names = next(iterator)
            for name in names:
                digest.update(str(name).encode("utf-8"))
                digest.update(b"\0")
            for key in ("image", "label", "input_ids", "attention_mask"):
                digest.update(
                    batch[key].detach().cpu().contiguous().numpy().tobytes()
                )
    finally:
        del iterator
        gc.collect()
    return digest.hexdigest()


def fresh_generators():
    return (
        make_generator(config.sampler_seed),
        make_generator(config.worker_seed),
    )


def check_text_dropout_rng_isolation(dataset):
    input_ids = dataset.input_ids[:16].clone()
    attention_mask = dataset.attention_masks[:16].clone()
    baseline_generator = make_generator(config.text_modality_dropout_seed)
    baseline_state = baseline_generator.get_state().clone()
    baseline_ids, baseline_mask, baseline_dropped = (
        apply_text_modality_dropout(
            input_ids,
            attention_mask,
            probability=0.0,
            generator=baseline_generator,
            neutral_input_ids=dataset.neutral_input_ids,
            neutral_attention_mask=dataset.neutral_attention_mask,
        )
    )
    if (
        baseline_dropped != 0
        or not torch.equal(baseline_ids, input_ids)
        or not torch.equal(baseline_mask, attention_mask)
        or not torch.equal(baseline_state, baseline_generator.get_state())
    ):
        raise RuntimeError("p=0 text dropout must be an exact no-op")

    generator_a = make_generator(config.text_modality_dropout_seed)
    output_a = apply_text_modality_dropout(
        input_ids,
        attention_mask,
        probability=0.5,
        generator=generator_a,
        neutral_input_ids=dataset.neutral_input_ids,
        neutral_attention_mask=dataset.neutral_attention_mask,
    )
    torch.rand(50_009)
    generator_b = make_generator(config.text_modality_dropout_seed)
    output_b = apply_text_modality_dropout(
        input_ids,
        attention_mask,
        probability=0.5,
        generator=generator_b,
        neutral_input_ids=dataset.neutral_input_ids,
        neutral_attention_mask=dataset.neutral_attention_mask,
    )
    if (
        output_a[2] != output_b[2]
        or not torch.equal(output_a[0], output_b[0])
        or not torch.equal(output_a[1], output_b[1])
    ):
        raise RuntimeError("Text dropout depends on the global model RNG")
    return output_a[2]


def main():
    configure_determinism(config.seed)
    if config.split_protocol != GROUPED_PROTOCOL:
        raise RuntimeError("Grouped protocol must be active")
    if config.persistent_workers:
        raise RuntimeError("persistent_workers must be disabled")
    dataset = build_dataset()
    text_dropout_count = check_text_dropout_rng_isolation(dataset)

    sampler_a, worker_a = fresh_generators()
    digest_a = collect_prefix(build_loader(dataset, sampler_a, worker_a))

    # Deliberately consume the global model RNG. Dedicated loader streams must
    # keep sample order and augmentations unchanged.
    torch.rand(100_003)
    sampler_b, worker_b = fresh_generators()
    digest_b = collect_prefix(build_loader(dataset, sampler_b, worker_b))
    if digest_a != digest_b:
        raise RuntimeError(
            "Loader output changed after unrelated global RNG consumption"
        )

    sampler_continuous, worker_continuous = fresh_generators()
    continuous_loader = build_loader(
        dataset,
        sampler_continuous,
        worker_continuous,
    )
    epoch_one_digest = collect_prefix(continuous_loader)
    sampler_checkpoint_state = sampler_continuous.get_state()
    worker_checkpoint_state = worker_continuous.get_state()
    continuous_epoch_two = collect_prefix(continuous_loader)

    sampler_resumed, worker_resumed = fresh_generators()
    sampler_resumed.set_state(sampler_checkpoint_state)
    worker_resumed.set_state(worker_checkpoint_state)
    resumed_epoch_two = collect_prefix(
        build_loader(dataset, sampler_resumed, worker_resumed)
    )
    if continuous_epoch_two != resumed_epoch_two:
        raise RuntimeError(
            "Epoch-boundary loader resume does not reproduce epoch two"
        )
    print("samples={}".format(len(dataset)))
    print("global_rng_independence_sha256={}".format(digest_a))
    print("epoch_one_prefix_sha256={}".format(epoch_one_digest))
    print("epoch_two_prefix_sha256={}".format(continuous_epoch_two))
    print("text_dropout_fixed_batch_count={}".format(text_dropout_count))
    print("Deterministic loader/augmentation/resume harness passed.")


if __name__ == "__main__":
    main()

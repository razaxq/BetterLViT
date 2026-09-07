"""NVIDIA real-data launch gate for deterministic grouped V4-B training."""

import hashlib
import os
from pathlib import Path
import sys

import torch
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import Config as config  # noqa: E402
from Load_Dataset import ImageToImage2D, RandomGenerator  # noqa: E402
from nets.BetterLViT import BetterLViT  # noqa: E402
from reproducibility import (  # noqa: E402
    configure_determinism,
    reset_global_seed,
)
from split_protocol import (  # noqa: E402
    EXPECTED_TRAIN_SAMPLES,
    EXPECTED_VALIDATION_SAMPLES,
    GROUPED_PROTOCOL,
    load_grouped_manifest,
)
from utils import WeightedDiceFocal, read_text  # noqa: E402


def build_model():
    reset_global_seed(config.model_seed)
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


def require_gradient(name, parameter):
    gradient = parameter.grad
    if (
        gradient is None
        or not torch.isfinite(gradient).all()
        or gradient.abs().max().item() == 0.0
    ):
        raise RuntimeError("{} gradient is invalid".format(name))


def parameter_sha256(model):
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_configuration():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA GPU must be visible")
    if config.experiment_architecture_version != "fam_eppa_v4b":
        raise RuntimeError("Active architecture must be fam_eppa_v4b")
    if config.split_protocol != GROUPED_PROTOCOL:
        raise RuntimeError("Grouped sensitivity split must be active")
    if config.boundary_loss_weight != 0.0:
        raise RuntimeError("Boundary loss must remain disabled")
    if (
        config.loss_name != "dice_focal"
        or config.dice_loss_weight != 0.5
        or config.focal_loss_weight != 0.5
        or config.focal_gamma != 2.0
    ):
        raise RuntimeError("Objective must remain Dice/Focal 0.5/0.5 gamma=2")
    if config.resume_path:
        raise RuntimeError("The first grouped V4-B run must start from scratch")
    if config.text_modality_dropout_prob != 0.0:
        raise RuntimeError("Grouped V4-B rebaseline must use text dropout p=0.0")
    if config.batch_size != 16 or config.epochs != 200:
        raise RuntimeError("Formal run must use batch=16 and epochs=200")
    if config.persistent_workers:
        raise RuntimeError("persistent_workers must remain disabled")


def load_real_batch():
    grouped = load_grouped_manifest(
        config.split_manifest_path,
        config.dataset_root,
        config.split_protocol,
    )
    if (
        len(grouped["train_records"]) != EXPECTED_TRAIN_SAMPLES
        or len(grouped["validation_records"])
        != EXPECTED_VALIDATION_SAMPLES
    ):
        raise RuntimeError("Grouped split sample counts are invalid")
    text = read_text(
        os.path.join(config.task_dataset, "Train_Val_text.xlsx")
    )
    dataset = ImageToImage2D(
        config.train_dataset,
        config.task_name,
        text,
        transforms.Compose(
            [RandomGenerator([config.img_size, config.img_size])]
        ),
        image_size=config.img_size,
        sample_records=grouped["train_records"],
    )
    if len(dataset) != EXPECTED_TRAIN_SAMPLES:
        raise RuntimeError("Real grouped training dataset must contain 5716")
    samples = [dataset[index][0] for index in range(config.batch_size)]
    return {
        key: torch.stack([sample[key] for sample in samples]).cuda()
        for key in ("image", "label", "input_ids", "attention_mask")
    }, grouped


def validate_structure(model):
    for stage in ("up4", "up3"):
        module = getattr(model, stage).eppa
        if not module.use_adaptive_frequency:
            raise RuntimeError("{} must enable adaptive frequency".format(stage))
    for stage in ("up2", "up1"):
        if getattr(model, stage).eppa.use_adaptive_frequency:
            raise RuntimeError("{} must retain V4-A behavior".format(stage))


def main():
    configure_determinism(config.seed)
    validate_configuration()
    batch, grouped = load_real_batch()
    model = build_model().cuda().train()
    validate_structure(model)
    initial_parameter_sha256 = parameter_sha256(model)
    reset_global_seed(config.training_seed)
    criterion = WeightedDiceFocal(
        dice_weight=config.dice_loss_weight,
        focal_weight=config.focal_loss_weight,
        focal_gamma=config.focal_gamma,
        focal_positive_weight=config.focal_positive_weight,
        focal_negative_weight=config.focal_negative_weight,
    )
    torch.cuda.reset_peak_memory_stats()
    predictions = model(
        batch["image"],
        batch["input_ids"],
        batch["attention_mask"],
    )
    loss = criterion(predictions, batch["label"].float())
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError("Dice/Focal objective produced a non-finite loss")
    if not torch.isfinite(predictions).all():
        raise RuntimeError("Model produced non-finite predictions")
    require_gradient("output head", model.outc.weight)
    for stage in ("up4", "up3"):
        adaptive = getattr(model, stage).eppa.adaptive_frequency
        require_gradient(
            "{} ALPF predictor".format(stage),
            adaptive.low_kernel_predictor.weight,
        )
        require_gradient(
            "{} AHPF predictor".format(stage),
            adaptive.high_kernel_predictor.weight,
        )
        require_gradient(
            "{} ALPF strength".format(stage),
            adaptive.alpf_strength_logit,
        )
        require_gradient(
            "{} AHPF strength".format(stage),
            adaptive.ahpf_strength_logit,
        )
    torch.cuda.synchronize()
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)
    print("Batch:", config.batch_size)
    print("Prediction:", tuple(predictions.shape))
    print("Loss: {:.6f}".format(loss.item()))
    print("Components:", criterion.last_components)
    print("Initial parameter SHA256:", initial_parameter_sha256)
    print("Manifest SHA256:", grouped["manifest_sha256"])
    print(
        "Peak GPU memory: {:.2f} GiB".format(
            torch.cuda.max_memory_allocated() / 1024 ** 3
        )
    )
    print("Deterministic grouped V4-B launch gate passed.")


if __name__ == "__main__":
    main()

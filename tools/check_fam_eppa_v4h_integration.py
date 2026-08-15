"""GPU and real-data integration check for FAM-EPPA V4-H."""

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
from utils import WeightedDiceFocal, read_text  # noqa: E402


def build_model():
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


def load_real_batch():
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
    )
    samples = [dataset[index][0] for index in range(config.batch_size)]
    return {
        key: torch.stack([sample[key] for sample in samples]).cuda()
        for key in ("image", "label", "input_ids", "attention_mask")
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-compatible PyTorch environment is required")
    if config.experiment_architecture_version != "fam_eppa_v4h":
        raise RuntimeError("The active configuration is not FAM-EPPA V4-H")
    if config.boundary_loss_weight != 0.0:
        raise RuntimeError("Boundary loss must remain disabled")
    if config.loss_name != "dice_focal" or config.resume_path:
        raise RuntimeError("V4-H must start with Dice/Focal from scratch")

    batch = load_real_batch()
    model = build_model().cuda().train()
    for stage in ("up4", "up3"):
        module = getattr(model, stage).eppa
        if not module.use_adaptive_frequency:
            raise RuntimeError("{} must retain adaptive frequency".format(stage))
        if not module.use_text_frequency_routing:
            raise RuntimeError("{} must enable text frequency".format(stage))
        if (
            module.use_plam_calibration
            or module.use_token_routing
            or module.use_semantic_flow_alignment
            or module.use_semantic_prototype_aggregation
            or module.use_semantic_reliability_residual
        ):
            raise RuntimeError("{} must disable V4-C through V4-G".format(stage))
    for stage in ("up2", "up1"):
        module = getattr(model, stage).eppa
        if module.use_adaptive_frequency or module.use_text_frequency_routing:
            raise RuntimeError("{} must retain V4-A behavior".format(stage))

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

    require_gradient("output head", model.outc.weight)
    for stage in ("up4", "up3"):
        route_head = getattr(model, stage).eppa.adaptive_frequency
        require_gradient(
            "{} text-frequency route head".format(stage),
            route_head.text_frequency_route_predictor.weight,
        )

    torch.cuda.synchronize()
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA/ROCm:", torch.version.cuda, torch.version.hip)
    print("Batch:", config.batch_size)
    print("Prediction:", tuple(predictions.shape))
    print("Loss: {:.6f}".format(loss.item()))
    print("Components:", criterion.last_components)
    print(
        "Peak GPU memory: {:.2f} GiB".format(
            torch.cuda.max_memory_allocated() / 1024 ** 3
        )
    )
    print("FAM-EPPA V4-H real-data integration check passed.")


if __name__ == "__main__":
    main()

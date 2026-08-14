"""GPU and real-data integration check for FAM-EPPA V4-F."""

import math
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
    if config.experiment_architecture_version != "fam_eppa_v4f":
        raise RuntimeError("The active configuration is not FAM-EPPA V4-F")
    if config.boundary_loss_weight != 0.0:
        raise RuntimeError("Boundary loss must remain disabled")
    if config.loss_name != "dice_focal" or config.resume_path:
        raise RuntimeError("V4-F must start with Dice/Focal from scratch")

    batch = load_real_batch()
    model = build_model().cuda().train()
    up4 = model.up4.eppa
    if not up4.use_adaptive_frequency:
        raise RuntimeError("up4 must retain adaptive frequency")
    if not up4.use_semantic_prototype_aggregation:
        raise RuntimeError("up4 must enable semantic prototypes")
    if (
        up4.use_plam_calibration
        or up4.use_token_routing
        or up4.use_semantic_flow_alignment
    ):
        raise RuntimeError("up4 must disable V4-C, V4-D and V4-E branches")
    up4.semantic_prototype_strength_raw.data.fill_(
        math.atanh(0.05 / up4.semantic_prototype_strength_max)
    )
    # The production region output is zero-initialized for V4-B compatibility.
    # Open it only in this reachability test so prototype gradients are
    # observable before the normal first optimizer step trains region_out.
    torch.nn.init.normal_(up4.region_out[-1].weight, std=1e-3)

    up3 = model.up3.eppa
    if (
        not up3.use_adaptive_frequency
        or up3.use_semantic_prototype_aggregation
    ):
        raise RuntimeError("up3 must retain the unmodified V4-B path")
    for stage in ("up2", "up1"):
        module = getattr(model, stage).eppa
        if (
            module.use_adaptive_frequency
            or module.use_semantic_prototype_aggregation
            or module.use_plam_calibration
            or module.use_token_routing
            or module.use_semantic_flow_alignment
        ):
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
    require_gradient(
        "up4 prototype assignment",
        up4.semantic_prototype_assign.weight,
    )
    require_gradient(
        "up4 prototype strength",
        up4.semantic_prototype_strength_raw,
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
    print("FAM-EPPA V4-F real-data integration check passed.")


if __name__ == "__main__":
    main()

"""AMD GPU and real-data integration check for FAM-EPPA V4-B."""

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
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("An AMD ROCm PyTorch environment is required.")
    if config.experiment_architecture_version != "fam_eppa_v4b":
        raise RuntimeError("The active configuration is not FAM-EPPA V4-B.")
    if config.boundary_loss_weight != 0.0:
        raise RuntimeError("Boundary loss must remain disabled.")
    if config.loss_name != "dice_focal":
        raise RuntimeError("V4-B must use Dice/Focal loss.")
    if config.resume_path:
        raise RuntimeError("V4-B must start from scratch.")

    batch = load_real_batch()
    model = build_model().cuda().train()
    for stage in ("up4", "up3"):
        module = getattr(model, stage).eppa
        if not module.use_adaptive_frequency:
            raise RuntimeError("{} must enable adaptive frequency".format(stage))
    for stage in ("up2", "up1"):
        module = getattr(model, stage).eppa
        if module.use_adaptive_frequency:
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
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    adaptive_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".adaptive_frequency." in name
    )
    print("GPU:", torch.cuda.get_device_name(0))
    print("ROCm:", torch.version.hip)
    print("Batch:", config.batch_size)
    print("Prediction:", tuple(predictions.shape))
    print("Loss: {:.6f}".format(loss.item()))
    print("Components:", criterion.last_components)
    print("Trainable parameters: {:,}".format(trainable_parameters))
    print("Adaptive frequency parameters: {:,}".format(adaptive_parameters))
    print(
        "Peak GPU memory: {:.2f} GiB".format(
            torch.cuda.max_memory_allocated() / 1024 ** 3
        )
    )
    print("FAM-EPPA V4-B real-data integration check passed.")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == "nt":
        os._exit(0)

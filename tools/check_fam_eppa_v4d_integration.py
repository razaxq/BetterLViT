"""CUDA and real-data integration check for FAM-EPPA V4-D."""

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
    if config.experiment_architecture_version != "fam_eppa_v4d":
        raise RuntimeError("The active configuration is not FAM-EPPA V4-D")
    if config.boundary_loss_weight != 0.0:
        raise RuntimeError("Boundary loss must remain disabled")
    if config.loss_name != "dice_focal" or config.resume_path:
        raise RuntimeError("V4-D must start with Dice/Focal from scratch")

    batch = load_real_batch()
    model = build_model().cuda().train()
    for stage in ("up4", "up3"):
        module = getattr(model, stage).eppa
        if not module.use_adaptive_frequency:
            raise RuntimeError("{} must enable frequency routing".format(stage))
        if not module.use_token_routing:
            raise RuntimeError("{} must enable token routing".format(stage))
        if module.use_semantic_flow_alignment:
            raise RuntimeError("{} must disable semantic flow".format(stage))
        # Move beyond safe initialization so this one-step integration check
        # can verify gradients throughout the token branch.
        torch.nn.init.normal_(module.token_out.weight, std=1e-3)
        torch.nn.init.normal_(module.region_out[-1].weight, std=1e-3)
    for stage in ("up2", "up1"):
        module = getattr(model, stage).eppa
        if (
            module.use_adaptive_frequency
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
    for stage in ("up4", "up3"):
        module = getattr(model, stage).eppa
        require_gradient(
            "{} token query".format(stage),
            module.token_query.weight,
        )
        require_gradient(
            "{} token key".format(stage),
            module.token_key.weight,
        )
        require_gradient(
            "{} token value".format(stage),
            module.token_value.weight,
        )
        require_gradient(
            "{} token output".format(stage),
            module.token_out.weight,
        )

    torch.cuda.synchronize()
    print("GPU:", torch.cuda.get_device_name(0))
    print("ROCm:", torch.version.hip)
    print("Batch:", config.batch_size)
    print("Prediction:", tuple(predictions.shape))
    print("Loss: {:.6f}".format(loss.item()))
    print("Components:", criterion.last_components)
    print(
        "Peak GPU memory: {:.2f} GiB".format(
            torch.cuda.max_memory_allocated() / 1024 ** 3
        )
    )
    print("FAM-EPPA V4-D real-data integration check passed.")


if __name__ == "__main__":
    main()

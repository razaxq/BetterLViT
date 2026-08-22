"""Synthetic forward/backward validation for one paper ablation profile."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        required=True,
        choices=(
            "b0_baseline",
            "a0_lora",
            "a1_lora_focal",
            "a2_lora_freq",
            "a3_lora_fmiseg",
        ),
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    os.environ["BETTERLVIT_EXPERIMENT"] = args.experiment
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import Config as config
    from nets.BetterLViT import BetterLViT
    from utils import WeightedDiceBCE, WeightedDiceFocal

    if not args.cpu and (not torch.cuda.is_available() or not torch.version.hip):
        raise RuntimeError("An AMD ROCm PyTorch environment is required.")
    device = torch.device("cpu" if args.cpu else "cuda")

    model = BetterLViT(
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
    ).to(device)
    model.train()

    text_parameters = list(model.text_encoder.named_parameters())
    trainable_text = [name for name, value in text_parameters if value.requires_grad]
    lora_parameters = [name for name, _ in text_parameters if "lora_" in name]
    if config.text_use_lora:
        if not trainable_text or not lora_parameters:
            raise RuntimeError("LoRA profile has no trainable LoRA parameters.")
    else:
        if trainable_text:
            raise RuntimeError("B0 text encoder is not fully frozen.")
        if model.text_encoder.training:
            raise RuntimeError("B0 frozen text encoder must remain in eval mode.")

    images = torch.randn(
        args.batch_size,
        3,
        config.img_size,
        config.img_size,
        device=device,
    )
    input_ids = torch.zeros(
        args.batch_size,
        config.text_max_len,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(
        0,
        2,
        (args.batch_size, 1, config.img_size, config.img_size),
        dtype=torch.float32,
        device=device,
    )

    output = model(images, input_ids, attention_mask)
    if tuple(output.shape) != tuple(labels.shape):
        raise RuntimeError(
            "Unexpected output shape: {} != {}".format(
                tuple(output.shape),
                tuple(labels.shape),
            )
        )
    if not torch.isfinite(output).all():
        raise RuntimeError("Model output contains non-finite values.")

    if config.loss_name == "dice_focal":
        criterion = WeightedDiceFocal(
            dice_weight=config.dice_loss_weight,
            focal_weight=config.focal_loss_weight,
            focal_gamma=config.focal_gamma,
            focal_positive_weight=config.focal_positive_weight,
            focal_negative_weight=config.focal_negative_weight,
        )
    else:
        criterion = WeightedDiceBCE(
            dice_weight=config.dice_loss_weight,
            BCE_weight=1.0 - config.dice_loss_weight,
        )
    loss = criterion(output, labels)
    if not torch.isfinite(loss):
        raise RuntimeError("Loss is non-finite.")
    loss.backward()

    trainable_with_gradient = sum(
        1
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    )
    if trainable_with_gradient == 0:
        raise RuntimeError("No trainable parameter received a gradient.")

    result = {
        "experiment": config.experiment_name,
        "paper_id": config.experiment_paper_id,
        "architecture_version": config.experiment_architecture_version,
        "decoder_fusion_mode": config.decoder_fusion_mode,
        "loss_name": config.loss_name,
        "text_use_lora": config.text_use_lora,
        "device": str(device),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "output_shape": list(output.shape),
        "batch_size": args.batch_size,
        "max_memory_allocated_gb": round(
            torch.cuda.max_memory_allocated() / (1024 ** 3),
            3,
        ) if device.type == "cuda" else 0.0,
        "max_memory_reserved_gb": round(
            torch.cuda.max_memory_reserved() / (1024 ** 3),
            3,
        ) if device.type == "cuda" else 0.0,
        "loss": float(loss.detach().cpu()),
        "trainable_text_tensors": len(trainable_text),
        "lora_parameter_tensors": len(lora_parameters),
        "trainable_tensors_with_gradient": trainable_with_gradient,
        "status": "ok",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

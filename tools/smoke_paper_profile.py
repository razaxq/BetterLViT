"""Synthetic forward/backward validation for one paper ablation profile."""

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
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
            "a4_lora_freq_focal",
            "a9_frozen_freq_focal",
            "a6_tcsr",
            "a7_tcsr_freq",
            "a8_tcsrv2_freq_focal",
            "p1_tcsrv21_boundary_router",
            "p2_tcsrv22_single_hop_boundary",
            "p3_tcsrv23_calibrated_gate",
            "p4_tcsrv24_sparse_boundary",
        ),
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1219)
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
    from train_model import build_optimizer_parameter_groups
    from utils import WeightedDiceBCE, WeightedDiceFocal

    torch.backends.cudnn.enabled = config.cudnn_enabled
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = config.deterministic_training
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(config.deterministic_training)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("A CUDA-capable PyTorch environment is required.")
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
            raise RuntimeError("Frozen text encoder has trainable parameters.")
        if model.text_encoder.training:
            raise RuntimeError("Frozen text encoder must remain in eval mode.")

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
    segmentation_loss = criterion(output, labels)
    router = getattr(model, "tcsr", None)
    router_regularization = segmentation_loss.new_zeros(())
    if router is not None and hasattr(router, "regularization_loss"):
        router_regularization = router.regularization_loss()
    loss = segmentation_loss + router_regularization
    if not torch.isfinite(loss):
        raise RuntimeError("Loss is non-finite.")
    loss.backward()

    optimizer_groups, _, _, router_names = build_optimizer_parameter_groups(
        model,
        config.weight_decay,
        config.learning_rate,
        config.tcsr_router_lr_scale,
    )
    router_group_lrs = sorted({
        float(group["lr"])
        for group in optimizer_groups
        if str(group.get("group_kind", "")).startswith("router_")
    })
    if config.tcsr_enabled and not router_names:
        raise RuntimeError("TCSR profile has no router optimizer parameters.")
    if config.tcsr_enabled and router_group_lrs != [
        config.learning_rate * config.tcsr_router_lr_scale
    ]:
        raise RuntimeError(
            "Unexpected router learning rates: {}".format(router_group_lrs)
        )

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
        "tcsr_enabled": config.tcsr_enabled,
        "tcsr_version": config.tcsr_version,
        "loss_name": config.loss_name,
        "text_use_lora": config.text_use_lora,
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "cudnn_enabled": config.cudnn_enabled,
        "deterministic_backend": config.deterministic_training,
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
        "segmentation_loss": float(segmentation_loss.detach().cpu()),
        "tcsr_regularization_loss": float(
            router_regularization.detach().cpu()
        ),
        "tcsr_stats": dict(getattr(router, "_last_stats", {}) or {}),
        "router_optimizer_parameter_tensors": len(router_names),
        "router_optimizer_learning_rates": router_group_lrs,
        "trainable_text_tensors": len(trainable_text),
        "lora_parameter_tensors": len(lora_parameters),
        "trainable_tensors_with_gradient": trainable_with_gradient,
        "status": "ok",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Run identity, balance, gradient, and AMD checks for BR-DG-EPPA."""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torchvision import transforms

import Config as config
from Load_Dataset import ImageToImage2D, RandomGenerator
from nets.BetterLViT import BetterLViT
from utils import WeightedDiceBCE, read_text


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


def require_finite_nonzero_gradient(name, parameter):
    gradient = parameter.grad
    if (
        gradient is None
        or not torch.isfinite(gradient).all()
        or gradient.abs().max().item() == 0
    ):
        raise RuntimeError(f"{name} gradient is invalid")


def check_identity_initialisation(model):
    module = model.up1.eppa.cuda().eval()
    skip = torch.randn(2, 64, 24, 24, device="cuda")
    decoder = torch.randn_like(skip)
    text = torch.randn(2, config.text_max_len, 768, device="cuda")
    with torch.inference_mode():
        refined = module(skip, decoder=decoder, text=text)
    maximum_error = (refined - skip).abs().max().item()
    if maximum_error > 1e-6:
        raise RuntimeError(
            "BR-DG-EPPA is not identity-initialised: "
            f"max error={maximum_error}"
        )
    stats = module._last_stats
    if stats is None:
        raise RuntimeError("BR-DG-EPPA did not expose evaluation statistics")
    if abs(stats["spatial_local_mean"]) > 1e-6:
        raise RuntimeError(
            "Balanced spatial residual is not zero mean: "
            f"{stats['spatial_local_mean']}"
        )
    if abs(stats["spatial_mean"] - 1.0) > 1e-6:
        raise RuntimeError(
            "Initial spatial gain is not identity: "
            f"{stats['spatial_mean']}"
        )
    module.train()
    return maximum_error, stats


def check_balanced_invariants(model):
    module = model.up1.eppa.cuda().eval()
    skip = torch.randn(2, 64, 24, 24, device="cuda")
    decoder = torch.randn_like(skip)
    text = torch.randn(2, config.text_max_len, 768, device="cuda")
    with torch.inference_mode():
        module(skip, decoder=decoder, text=text)
    stats = module._last_stats
    if abs(stats["spatial_local_mean"]) > 1e-6:
        raise RuntimeError(
            "Trained local spatial residual lost its zero-mean invariant"
        )
    theoretical_minimum = (
        1.0
        - module.local_strength_max
        - module.global_strength_max
    )
    theoretical_maximum = (
        1.0
        + module.local_strength_max
        + module.global_strength_max
    )
    if (
        stats["spatial_min"] < theoretical_minimum - 1e-6
        or stats["spatial_max"] > theoretical_maximum + 1e-6
    ):
        raise RuntimeError(
            "Spatial gain escaped its bounded residual interval"
        )
    guide_weight_sum = sum(
        stats[key]
        for key in (
            "guide_skip_weight",
            "guide_decoder_weight",
            "guide_local_edge_weight",
            "guide_context_edge_weight",
        )
    )
    if abs(guide_weight_sum - 1.0) > 1e-6:
        raise RuntimeError(
            "Guide branch weights do not form a convex mixture"
        )
    module.train()
    return stats


def main():
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("An AMD ROCm PyTorch environment is required.")
    if config.boundary_loss_weight != 0.0:
        raise RuntimeError(
            "Architecture experiment must disable boundary loss."
        )

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
    samples = [
        dataset[index][0] for index in range(config.batch_size)
    ]
    images = torch.stack(
        [sample["image"] for sample in samples]
    ).cuda()
    labels = torch.stack(
        [sample["label"] for sample in samples]
    ).float().cuda()
    input_ids = torch.stack(
        [sample["input_ids"] for sample in samples]
    ).cuda()
    attention_mask = torch.stack(
        [sample["attention_mask"] for sample in samples]
    ).cuda()

    model = build_model().cuda().train()
    if any(
        name.startswith("pix_module")
        for name, _ in model.named_modules()
    ):
        raise RuntimeError("Dead PLAM modules are still registered.")
    identity_error, initial_stats = check_identity_initialisation(model)

    criterion = WeightedDiceBCE(
        dice_weight=0.5,
        BCE_weight=0.5,
        boundary_weight=config.boundary_loss_weight,
        boundary_kernel_size=config.boundary_kernel_size,
    )
    optimizer = torch.optim.Adam(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    torch.cuda.reset_peak_memory_stats()
    for step in range(1, 3):
        predictions = model(images, input_ids, attention_mask)
        loss = criterion(predictions, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        require_finite_nonzero_gradient(
            "output head",
            model.outc.weight,
        )
        require_finite_nonzero_gradient(
            "BR-DG-EPPA spatial head",
            model.up1.eppa.spatial_out.weight,
        )
        require_finite_nonzero_gradient(
            "BR-DG-EPPA channel head",
            model.up1.eppa.channel_mlp[-1].weight,
        )
        if step == 2:
            require_finite_nonzero_gradient(
                "BR-DG-EPPA decoder guide",
                model.up1.eppa.decoder_semantic_proj.weight,
            )
            require_finite_nonzero_gradient(
                "BR-DG-EPPA dilated edge branch",
                model.up1.eppa.edge_context.weight,
            )
            require_finite_nonzero_gradient(
                "BR-DG-EPPA guide branch weights",
                model.up1.eppa.guide_branch_logits,
            )
            require_finite_nonzero_gradient(
                "BR-DG-EPPA local strength",
                model.up1.eppa.local_strength_logit,
            )
            require_finite_nonzero_gradient(
                "BR-DG-EPPA global strength",
                model.up1.eppa.global_strength_logit,
            )
            require_finite_nonzero_gradient(
                "BR-DG-EPPA spatial text FiLM",
                model.up1.eppa.text_spatial_film.weight,
            )
        if not torch.isfinite(loss):
            raise RuntimeError("Objective produced a non-finite loss")
        optimizer.step()
        print(
            f"step={step} loss={loss.item():.6f} "
            f"components={criterion.last_components}",
            flush=True,
        )

    trained_stats = check_balanced_invariants(model)
    torch.cuda.synchronize()
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    eppa_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".eppa." in name
    )
    print("GPU:", torch.cuda.get_device_name(0))
    print("Batch:", config.batch_size)
    print("Prediction:", tuple(predictions.shape))
    print(f"Identity max error: {identity_error:.3e}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(f"BR-DG-EPPA parameters: {eppa_parameters:,}")
    print(
        "Initial residual strengths: "
        f"local={initial_stats['local_strength_mean']:.4f}, "
        f"global={initial_stats['global_strength_mean']:.4f}"
    )
    print(
        "Post-step spatial gain range: "
        f"[{trained_stats['spatial_min']:.4f}, "
        f"{trained_stats['spatial_max']:.4f}]"
    )
    print(
        "Peak GPU memory: "
        f"{torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GiB"
    )
    print("BR-DG-EPPA architecture check passed.")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == "nt":
        os._exit(0)

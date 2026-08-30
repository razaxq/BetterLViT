"""Deterministic CPU/GPU checks for Text-Conditioned Cross-Scale Skip Router."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nets.tcsr import TextConditionedCrossScaleSkipRouter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1219)
    return parser.parse_args()


def require_nonzero_finite_gradient(name, parameter):
    gradient = parameter.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError("{} has no finite gradient.".format(name))
    if gradient.abs().sum().item() == 0.0:
        raise RuntimeError("{} has an all-zero gradient.".format(name))


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device("cuda" if args.cuda else "cpu")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False

    channels = (8, 16, 32, 64)
    spatial_sizes = (32, 16, 8, 4)
    router = TextConditionedCrossScaleSkipRouter(
        channels,
        text_dim=16,
        routing_dim=8,
        max_residual_strength=1.0,
    ).to(device)
    router.eval()
    skips = tuple(
        torch.randn(
            args.batch_size,
            channel,
            size,
            size,
            device=device,
            requires_grad=True,
        )
        for channel, size in zip(channels, spatial_sizes)
    )
    text = torch.randn(
        args.batch_size,
        7,
        16,
        device=device,
        requires_grad=True,
    )
    text_mask = torch.ones(
        args.batch_size,
        7,
        dtype=torch.long,
        device=device,
    )
    text_mask[0, -2:] = 0
    if args.batch_size > 1:
        text_mask[1].zero_()

    identity_outputs = router(skips, text, text_mask=text_mask)
    identity_error = max(
        (output - source).abs().max().item()
        for output, source in zip(identity_outputs, skips)
    )
    if identity_error != 0.0:
        raise RuntimeError(
            "Identity initialization changed a skip: {:.3e}.".format(
                identity_error
            )
        )
    if any(tuple(output.shape) != tuple(source.shape)
           for output, source in zip(identity_outputs, skips)):
        raise RuntimeError("TCSR changed one or more skip shapes.")
    if not all(torch.isfinite(output).all() for output in identity_outputs):
        raise RuntimeError("Identity pass produced a non-finite output.")
    first_stats = dict(router._last_stats)
    if abs(first_stats["scale_weight_sum"] - 1.0) > 1e-6:
        raise RuntimeError("Scale routing weights do not sum to one.")

    identity_loss = sum(output.square().mean() for output in identity_outputs)
    identity_loss.backward()
    require_nonzero_finite_gradient("residual_gate", router.residual_gate)

    router.zero_grad(set_to_none=True)
    for source in skips:
        source.grad = None
    text.grad = None
    with torch.no_grad():
        router.residual_gate.fill_(0.25)
    active_outputs_1 = router(skips, text, text_mask=text_mask)
    active_outputs_2 = router(skips, text, text_mask=text_mask)
    deterministic_error = max(
        (left - right).abs().max().item()
        for left, right in zip(active_outputs_1, active_outputs_2)
    )
    if deterministic_error != 0.0:
        raise RuntimeError(
            "Repeated deterministic forwards differ: {:.3e}.".format(
                deterministic_error
            )
        )
    active_loss = sum(output.square().mean() for output in active_outputs_1)
    active_loss.backward()
    gradient_checks = {
        "residual_gate": router.residual_gate,
        "visual_projection": router.visual_projections[0][0].weight,
        "text_key": router.text_key.weight,
        "scale_score": router.scale_score_heads[0][-1].weight,
        "channel_head": router.channel_heads[0].weight,
        "spatial_head": router.spatial_local[0].weight,
    }
    for name, parameter in gradient_checks.items():
        require_nonzero_finite_gradient(name, parameter)

    active_stats = dict(router._last_stats)
    result = {
        "status": "ok",
        "architecture_version": router.architecture_version,
        "device": str(device),
        "batch_size": args.batch_size,
        "parameter_count": sum(
            parameter.numel() for parameter in router.parameters()
        ),
        "identity_max_abs_error": identity_error,
        "deterministic_repeat_max_abs_error": deterministic_error,
        "scale_weights": active_stats["scale_weights"],
        "scale_weight_sum": active_stats["scale_weight_sum"],
        "effective_gates": active_stats["effective_gates"],
        "all_required_gradients_nonzero": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Deterministic behavioral checks for boundary-free CDRR V1."""

import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nets.cdrr import CDRRRefiner
from utils import DualHeadMaskObjective


def gradient_sum(parameters):
    return float(sum(
        parameter.grad.detach().abs().sum().item()
        for parameter in parameters
        if parameter.grad is not None
    ))


def main():
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(1219)
    module = CDRRRefiner(
        channels=64,
        hidden_channels=32,
        delta_max=0.5,
        active_fraction=0.15,
    ).train()
    coarse_feature = torch.randn(
        2, 64, 112, 112, requires_grad=True
    )
    fine_feature = torch.randn(
        2, 64, 224, 224, requires_grad=True
    )
    base_logits = torch.randn(
        2, 1, 224, 224, requires_grad=True
    )
    labels = torch.randint(0, 2, (2, 1, 224, 224)).float()

    first = module(coarse_feature, fine_feature, base_logits)
    repeated = module(coarse_feature, fine_feature, base_logits)
    identity_error = (first["final"] - first["base"]).abs().max()
    repeat_error = (first["final"] - repeated["final"]).abs().max()
    outside_error = first["delta"][first["support"] == 0].abs().max()
    support_fraction = first["support"].mean()
    if identity_error.item() != 0.0 or repeat_error.item() != 0.0:
        raise RuntimeError("CDRR zero-init identity/determinism failed")
    if outside_error.item() != 0.0:
        raise RuntimeError("CDRR changed pixels outside its support")
    if abs(support_fraction.item() - 0.15) > 1e-4:
        raise RuntimeError("CDRR support fraction is not locked")

    objective = DualHeadMaskObjective(aux_weight=0.1)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    first_loss = objective(first, labels)
    first_loss.backward()
    projection_gradient = gradient_sum(module.refiner[-1].parameters())
    coarse_gradient = gradient_sum(module.coarse_head.parameters())
    base_gradient = float(base_logits.grad.detach().abs().sum().item())
    if projection_gradient <= 0.0 or coarse_gradient <= 0.0:
        raise RuntimeError("CDRR heads did not receive gradients")
    if base_gradient <= 0.0:
        raise RuntimeError("CDRR final prediction did not train the base head")
    if coarse_feature.grad is not None or fine_feature.grad is not None:
        raise RuntimeError("CDRR auxiliary paths leaked into the decoder trunk")

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    base_logits.grad = None
    second = module(coarse_feature, fine_feature, base_logits)
    second_loss = objective(second, labels)
    second_loss.backward()
    trunk_gradient = gradient_sum(module.refiner[:-1].parameters())
    active_delta = second["delta"][second["support"] > 0]
    positive_fraction = (active_delta > 0).float().mean()
    negative_fraction = (active_delta < 0).float().mean()
    if trunk_gradient <= 0.0:
        raise RuntimeError("CDRR refiner trunk did not receive gradients")
    if positive_fraction.item() <= 0.0 or negative_fraction.item() <= 0.0:
        raise RuntimeError("CDRR centered residual lost one correction direction")
    if second["delta"][second["support"] == 0].abs().max().item() != 0.0:
        raise RuntimeError("CDRR second-step residual leaked outside support")

    print(json.dumps({
        "architecture_version": module.architecture_version,
        "identity_max_abs_error": float(identity_error.item()),
        "repeat_max_abs_error": float(repeat_error.item()),
        "support_fraction": float(support_fraction.item()),
        "outside_support_max_abs_delta": float(outside_error.item()),
        "first_loss": float(first_loss.detach().item()),
        "second_loss": float(second_loss.detach().item()),
        "projection_gradient_l1": projection_gradient,
        "coarse_head_gradient_l1": coarse_gradient,
        "base_logit_gradient_l1": base_gradient,
        "second_step_refiner_gradient_l1": trunk_gradient,
        "decoder_feature_gradient_isolated": True,
        "positive_active_fraction": float(positive_fraction.item()),
        "negative_active_fraction": float(negative_fraction.item()),
        "boundary_target_used": False,
        "status": "ok",
    }, indent=2))


if __name__ == "__main__":
    main()

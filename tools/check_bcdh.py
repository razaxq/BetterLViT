"""Deterministic unit checks for the boundary-free BCDH-R V1 module."""

import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nets.bcdh import BCDHRefiner
from utils import BCDHObjective


def gradient_sum(parameters):
    return float(sum(
        parameter.grad.detach().abs().sum().item()
        for parameter in parameters
        if parameter.grad is not None
    ))


def main():
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(1219)
    module = BCDHRefiner(
        channels=64,
        hidden_channels=32,
        delta_max=1.0,
        detach_cues=True,
    ).train()
    coarse_feature = torch.randn(2, 64, 112, 112)
    fine_feature = torch.randn(2, 64, 224, 224)
    base_logits = torch.randn(2, 1, 224, 224, requires_grad=True)
    labels = torch.randint(0, 2, (2, 1, 224, 224)).float()

    first = module(coarse_feature, fine_feature, base_logits)
    repeated = module(coarse_feature, fine_feature, base_logits)
    identity_error = (first["final"] - first["base"]).abs().max()
    repeat_error = (first["final"] - repeated["final"]).abs().max()
    if identity_error.item() != 0.0 or repeat_error.item() != 0.0:
        raise RuntimeError("BCDH zero-init identity/determinism check failed")
    if first["coarse"].shape != first["final"].shape:
        raise RuntimeError("BCDH coarse prediction was not upsampled")

    objective = BCDHObjective(aux_weight=0.2)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    loss = objective(first, labels)
    loss.backward()
    final_projection_gradient = gradient_sum(module.refiner[-1].parameters())
    coarse_gradient = gradient_sum(module.coarse_head.parameters())
    if final_projection_gradient <= 0.0 or coarse_gradient <= 0.0:
        raise RuntimeError("BCDH output/coarse heads did not receive gradients")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    base_logits.grad = None

    second = module(coarse_feature, fine_feature, base_logits)
    second_loss = objective(second, labels)
    second_loss.backward()
    upstream_gradient = gradient_sum(module.refiner[:-1].parameters())
    if upstream_gradient <= 0.0:
        raise RuntimeError("BCDH refiner trunk did not receive second-step gradients")

    result = {
        "architecture_version": module.architecture_version,
        "identity_max_abs_error": float(identity_error.item()),
        "repeat_max_abs_error": float(repeat_error.item()),
        "first_loss": float(loss.detach().item()),
        "second_loss": float(second_loss.detach().item()),
        "final_projection_gradient_l1": final_projection_gradient,
        "coarse_head_gradient_l1": coarse_gradient,
        "second_step_trunk_gradient_l1": upstream_gradient,
        "boundary_target_used": False,
        "status": "ok",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

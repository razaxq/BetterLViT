"""CPU behavioral checks for RACE-Fuse and its report parser."""

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nets.race_fuse import RACEFuse  # noqa: E402
from race_semantics import make_zone_basis, parse_report_slots  # noqa: E402


def main():
    torch.manual_seed(1219)
    module = RACEFuse(channels=(8, 16, 24, 32), hidden_channels=8)
    skips = tuple(
        torch.randn(2, channels, 24 // (2 ** index), 24 // (2 ** index))
        for index, channels in enumerate((8, 16, 24, 32))
    )
    text = torch.randn(2, 12, 768)
    mask = torch.ones(2, 12, dtype=torch.long)
    basis = torch.from_numpy(make_zone_basis(24, 24)).repeat(2, 1, 1, 1)
    routed_a, aux_a = module(skips, text, mask, basis)
    routed_b, aux_b = module(skips, text, mask, basis)
    identity_error = max(
        float((before - after).detach().abs().max())
        for before, after in zip(skips, routed_a)
    )
    repeat_error = max(
        float((first - second).detach().abs().max())
        for first, second in zip(routed_a, routed_b)
    )
    if identity_error != 0.0 or repeat_error != 0.0:
        raise RuntimeError("RACE identity or determinism check failed")
    if aux_a["slot_logits"].shape != (2, 9):
        raise RuntimeError("Unexpected RACE slot shape")
    if not torch.equal(aux_a["slot_logits"], aux_b["slot_logits"]):
        raise RuntimeError("RACE slot head is not deterministic")

    bilateral = parse_report_slots(
        "two peripheral opacities in the bilateral lower lung zones"
    )
    unilateral = parse_report_slots("one opacity in the left upper lung")
    if bilateral[:6].tolist() != [0, 0, 1, 0, 0, 1]:
        raise RuntimeError("Bilateral lower-zone parsing failed")
    if unilateral[:6].tolist() != [1, 0, 0, 0, 0, 0]:
        raise RuntimeError("Left upper-zone parsing failed")
    if bilateral[7].item() != 1.0 or unilateral[6].item() != 1.0:
        raise RuntimeError("Count parsing failed")

    loss = sum(value.mean() for value in routed_a)
    loss.backward()
    if module.routes[0].strength_logit.grad is None:
        raise RuntimeError("RACE route strength received no gradient")
    print(json.dumps({
        "status": "ok",
        "identity_max_abs_error": identity_error,
        "repeat_max_abs_error": repeat_error,
        "route_strengths": module._last_stats["route_strengths"],
    }, indent=2))


if __name__ == "__main__":
    main()

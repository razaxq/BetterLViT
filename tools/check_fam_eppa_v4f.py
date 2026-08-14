"""Fast structural checks for FAM-EPPA V4-F semantic prototypes."""

import math
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import Config as config  # noqa: E402
from nets.eppa import FAMSemanticPrototypeEPPA  # noqa: E402


def build_block(use_prototypes):
    return FAMSemanticPrototypeEPPA(
        64,
        text_dim=768,
        min_bottleneck_channels=32,
        use_adaptive_frequency=True,
        frequency_groups=8,
        use_semantic_flow_alignment=False,
        use_token_routing=False,
        use_plam_calibration=False,
        use_semantic_prototype_aggregation=use_prototypes,
        semantic_prototype_count=4,
        semantic_prototype_temperature=0.75,
        semantic_prototype_strength_max=0.25,
        semantic_prototype_strength_init=0.0,
    )


def sample_inputs():
    return {
        "skip": torch.randn(2, 64, 24, 24),
        "plam": torch.randn(2, 64, 24, 24),
        "decoder": torch.randn(2, 64, 24, 24),
        "text": torch.randn(2, 32, 768),
        "text_mask": torch.ones(2, 32, dtype=torch.long),
    }


def check_config():
    assert config.experiment_architecture_version == "fam_eppa_v4f"
    assert config.boundary_loss_weight == 0.0
    assert config.loss_name == "dice_focal"
    assert config.resume_path == ""
    vit_config = config.get_CTranS_config()
    assert tuple(vit_config.eppa_adaptive_frequency_stages) == (
        "up4",
        "up3",
    )
    assert tuple(vit_config.eppa_semantic_flow_stages) == ()
    assert tuple(vit_config.eppa_token_routing_stages) == ()
    assert tuple(vit_config.eppa_plam_calibration_stages) == ()
    assert tuple(vit_config.eppa_semantic_prototype_stages) == ("up4",)
    assert vit_config.eppa_semantic_prototype_count == 4


def check_identity_initialization():
    torch.manual_seed(1228)
    prototype = build_block(use_prototypes=True).eval()
    baseline = build_block(use_prototypes=False).eval()
    incompatible = baseline.load_state_dict(
        prototype.state_dict(),
        strict=False,
    )
    assert not incompatible.missing_keys
    assert all(
        "semantic_prototype" in name
        for name in incompatible.unexpected_keys
    )
    inputs = sample_inputs()
    with torch.no_grad():
        prototype_output, prototype_decoder = prototype(
            **inputs,
            return_decoder=True,
        )
        baseline_output, baseline_decoder = baseline(
            **inputs,
            return_decoder=True,
        )
    if not torch.equal(prototype_decoder, baseline_decoder):
        raise AssertionError("V4-F must preserve the V4-B decoder path")
    if not torch.equal(prototype_output, baseline_output):
        raise AssertionError(
            "Zero prototype strength must exactly reproduce V4-B"
        )


def check_mean_preservation_and_gradients():
    torch.manual_seed(1229)
    block = build_block(use_prototypes=True).train()
    block.semantic_prototype_strength_raw.data.fill_(
        math.atanh(0.05 / block.semantic_prototype_strength_max)
    )
    semantic = torch.randn(2, 16, 24, 24, requires_grad=True)
    aggregated, diagnostics = block._semantic_prototype_aggregate(semantic)
    mean_error = (
        diagnostics["reconstruction"].mean(dim=(2, 3))
        - semantic.mean(dim=(2, 3))
    ).abs().amax().item()
    if mean_error >= 1e-5:
        raise AssertionError("Prototype reconstruction did not preserve mean")
    aggregated.square().mean().backward()
    for name in (
        "semantic_prototype_assign.weight",
        "semantic_prototype_strength_raw",
    ):
        gradient = dict(block.named_parameters())[name].grad
        if (
            gradient is None
            or not torch.isfinite(gradient).all()
            or gradient.abs().max().item() == 0.0
        ):
            raise AssertionError("Invalid gradient for {}".format(name))


def check_forward_backward_and_stats():
    torch.manual_seed(1230)
    block = build_block(use_prototypes=True).train()
    block.semantic_prototype_strength_raw.data.fill_(0.1)
    inputs = sample_inputs()
    for name in ("skip", "plam", "decoder", "text"):
        inputs[name].requires_grad_(True)
    output, decoder = block(**inputs, return_decoder=True)
    assert output.shape == inputs["skip"].shape
    assert decoder.shape == inputs["decoder"].shape
    assert torch.isfinite(output).all() and torch.isfinite(decoder).all()
    (output.square().mean() + decoder.square().mean()).backward()

    block.eval()
    with torch.no_grad():
        block(**{name: value.detach() for name, value in inputs.items()})
    stats = block._last_stats
    assert stats["architecture_version"] == "fam_eppa_v4f"
    assert stats["adaptive_frequency_enabled"] is True
    assert stats["semantic_prototype_enabled"] is True
    assert stats["semantic_flow_enabled"] is False
    assert stats["token_routing_enabled"] is False
    assert stats["plam_calibration_enabled"] is False
    assert stats["haar_reconstruction_error"] < 1e-5
    assert stats["semantic_prototype_mean_error"] < 1e-5
    assert abs(stats["alpf_kernel_sum"] - 1.0) < 1e-6
    assert abs(stats["ahpf_kernel_sum"] - 1.0) < 1e-6
    assert stats["semantic_prototype_active_ratio"] == 1.0
    for name, value in stats.items():
        if isinstance(value, float) and not torch.isfinite(torch.tensor(value)):
            raise AssertionError("Non-finite diagnostic: {}".format(name))


def main():
    check_config()
    check_identity_initialization()
    check_mean_preservation_and_gradients()
    check_forward_backward_and_stats()
    print("FAM-EPPA V4-F checks passed.")


if __name__ == "__main__":
    main()

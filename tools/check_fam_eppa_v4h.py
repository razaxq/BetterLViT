"""Fast structural checks for FAM-EPPA V4-H frequency routing."""

from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import Config as config  # noqa: E402
from nets.eppa import (  # noqa: E402
    FAMTokenConditionedFrequencyEPPA,
)


def build_block(use_text_frequency):
    return FAMTokenConditionedFrequencyEPPA(
        64,
        text_dim=768,
        min_bottleneck_channels=32,
        use_adaptive_frequency=True,
        frequency_groups=8,
        use_semantic_flow_alignment=False,
        use_token_routing=False,
        use_plam_calibration=False,
        use_semantic_prototype_aggregation=False,
        use_semantic_reliability_residual=False,
        use_text_frequency_routing=use_text_frequency,
        text_frequency_attention_dim=32,
        text_frequency_attention_heads=4,
        text_frequency_temperature_init=5.0,
        text_frequency_logit_max=1.0,
    )


def sample_inputs():
    text_mask = torch.ones(2, 32, dtype=torch.long)
    text_mask[0, 19:] = 0
    text_mask[1, 25:] = 0
    return {
        "skip": torch.randn(2, 64, 24, 24),
        "plam": torch.randn(2, 64, 24, 24),
        "decoder": torch.randn(2, 64, 24, 24),
        "text": torch.randn(2, 32, 768),
        "text_mask": text_mask,
    }


def require_gradient(name, parameter):
    gradient = parameter.grad
    if (
        gradient is None
        or not torch.isfinite(gradient).all()
        or gradient.abs().max().item() == 0.0
    ):
        raise AssertionError("Invalid gradient for {}".format(name))


def check_config():
    assert config.experiment_architecture_version == "fam_eppa_v4h"
    assert config.boundary_loss_weight == 0.0
    assert config.loss_name == "dice_focal"
    assert config.resume_path == ""
    vit_config = config.get_CTranS_config()
    assert tuple(vit_config.eppa_adaptive_frequency_stages) == (
        "up4",
        "up3",
    )
    assert tuple(vit_config.eppa_text_frequency_stages) == (
        "up4",
        "up3",
    )
    assert tuple(vit_config.eppa_semantic_flow_stages) == ()
    assert tuple(vit_config.eppa_token_routing_stages) == ()
    assert tuple(vit_config.eppa_plam_calibration_stages) == ()
    assert tuple(vit_config.eppa_semantic_prototype_stages) == ()
    assert tuple(vit_config.eppa_semantic_reliability_stages) == ()


def check_identity_initialization():
    torch.manual_seed(1233)
    routed = build_block(use_text_frequency=True).eval()
    baseline = build_block(use_text_frequency=False).eval()
    incompatible = baseline.load_state_dict(
        routed.state_dict(),
        strict=False,
    )
    assert not incompatible.missing_keys
    assert all(
        "text_frequency" in name
        for name in incompatible.unexpected_keys
    )
    inputs = sample_inputs()
    with torch.no_grad():
        routed_output, routed_decoder = routed(
            **inputs,
            return_decoder=True,
        )
        baseline_output, baseline_decoder = baseline(
            **inputs,
            return_decoder=True,
        )
    if not torch.equal(routed_decoder, baseline_decoder):
        raise AssertionError("V4-H must preserve V4-B decoder at init")
    if not torch.equal(routed_output, baseline_output):
        raise AssertionError("V4-H must exactly reproduce V4-B at init")


def check_gradient_path_and_stats():
    torch.manual_seed(1234)
    block = build_block(use_text_frequency=True).train()
    inputs = sample_inputs()
    for name in ("skip", "plam", "decoder", "text"):
        inputs[name].requires_grad_(True)

    output, decoder = block(**inputs, return_decoder=True)
    loss = output.square().mean() + decoder.square().mean()
    loss.backward()
    route_head = (
        block.adaptive_frequency.text_frequency_route_predictor
    )
    require_gradient("text-frequency route head", route_head.weight)

    # Move only the route head off zero. On the next backward pass the loss
    # must reach token query/key/value projections as well.
    with torch.no_grad():
        route_head.weight.add_(-0.1 * route_head.weight.grad)
        route_head.bias.add_(-0.1 * route_head.bias.grad)
    block.zero_grad(set_to_none=True)
    for value in inputs.values():
        if value.grad is not None:
            value.grad = None
    output, decoder = block(**inputs, return_decoder=True)
    (output.square().mean() + decoder.square().mean()).backward()
    for name in (
        "text_frequency_query.weight",
        "text_frequency_key.weight",
        "text_frequency_value.weight",
        "text_frequency_out.weight",
    ):
        require_gradient(
            name,
            dict(block.adaptive_frequency.named_parameters())[name],
        )

    block.eval()
    with torch.no_grad():
        block(**{name: value.detach() for name, value in inputs.items()})
    stats = block._last_stats
    assert stats["architecture_version"] == "fam_eppa_v4h"
    assert stats["adaptive_frequency_enabled"] is True
    assert stats["text_frequency_routing_enabled"] is True
    assert stats["token_routing_enabled"] is False
    assert stats["semantic_flow_enabled"] is False
    assert stats["semantic_reliability_enabled"] is False
    assert stats["haar_reconstruction_error"] < 1e-5
    assert abs(stats["alpf_kernel_sum"] - 1.0) < 1e-6
    assert abs(stats["ahpf_kernel_sum"] - 1.0) < 1e-6
    assert stats["text_frequency_route_delta_std"] > 0.0
    assert stats["text_frequency_low_weight_shift"] > 0.0
    assert stats["text_frequency_high_weight_shift"] > 0.0
    assert stats["text_frequency_valid_count_mean"] == 22.0
    for name, value in stats.items():
        if isinstance(value, float) and not torch.isfinite(torch.tensor(value)):
            raise AssertionError("Non-finite diagnostic: {}".format(name))


def main():
    check_config()
    check_identity_initialization()
    check_gradient_path_and_stats()
    print("FAM-EPPA V4-H checks passed.")


if __name__ == "__main__":
    main()

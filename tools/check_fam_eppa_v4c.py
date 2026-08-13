"""Fast structural checks for FAM-EPPA V4-C semantic-flow alignment."""

from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import Config as config  # noqa: E402
from nets.eppa import FAMSemanticFlowEPPA  # noqa: E402


def build_block(use_flow):
    return FAMSemanticFlowEPPA(
        64,
        text_dim=768,
        min_bottleneck_channels=32,
        use_adaptive_frequency=True,
        frequency_groups=8,
        use_semantic_flow_alignment=use_flow,
        flow_groups=4,
        flow_max_offset=1.5,
        flow_strength_init=0.25,
    )


def check_config():
    assert config.experiment_architecture_version == "fam_eppa_v4c"
    assert config.boundary_loss_weight == 0.0
    assert config.loss_name == "dice_focal"
    assert config.resume_path == ""
    assert config.require_checkpoint_architecture_match is True
    vit_config = config.get_CTranS_config()
    assert tuple(vit_config.eppa_adaptive_frequency_stages) == (
        "up4",
        "up3",
    )
    assert tuple(vit_config.eppa_semantic_flow_stages) == ("up4", "up3")
    assert vit_config.eppa_flow_groups == 4
    assert 0.0 < vit_config.eppa_flow_max_offset <= 2.0


def check_identity_initialization():
    torch.manual_seed(1221)
    aligned = build_block(use_flow=True).eval()
    baseline = build_block(use_flow=False).eval()
    incompatible = baseline.load_state_dict(
        aligned.state_dict(),
        strict=False,
    )
    assert not incompatible.missing_keys
    assert all("flow_" in name for name in incompatible.unexpected_keys)
    inputs = {
        "skip": torch.randn(2, 64, 24, 24),
        "plam": torch.randn(2, 64, 24, 24),
        "decoder": torch.randn(2, 64, 24, 24),
        "text": torch.randn(2, 32, 768),
    }
    with torch.no_grad():
        aligned_output, aligned_decoder = aligned(
            **inputs,
            return_decoder=True,
        )
        baseline_output, baseline_decoder = baseline(
            **inputs,
            return_decoder=True,
        )
    if not torch.allclose(aligned_decoder, baseline_decoder, atol=2e-5):
        raise AssertionError("Zero flow must reproduce the V4-B decoder")
    if not torch.allclose(aligned_output, baseline_output, atol=2e-5):
        raise AssertionError("Zero flow must reproduce the V4-B output")


def check_forward_backward_and_stats():
    torch.manual_seed(1222)
    block = build_block(use_flow=True).train()
    inputs = {
        "skip": torch.randn(2, 64, 24, 24, requires_grad=True),
        "plam": torch.randn(2, 64, 24, 24, requires_grad=True),
        "decoder": torch.randn(2, 64, 24, 24, requires_grad=True),
        "text": torch.randn(2, 32, 768, requires_grad=True),
    }
    output, decoder = block(**inputs, return_decoder=True)
    assert output.shape == inputs["skip"].shape
    assert decoder.shape == inputs["decoder"].shape
    assert torch.isfinite(output).all() and torch.isfinite(decoder).all()
    (output.square().mean() + decoder.square().mean()).backward()

    parameters = dict(block.named_parameters())
    required_gradients = (
        "adaptive_frequency.flow_predictor.3.weight",
        "adaptive_frequency.low_kernel_predictor.weight",
        "adaptive_frequency.high_kernel_predictor.weight",
    )
    for name in required_gradients:
        gradient = parameters[name].grad
        if (
            gradient is None
            or not torch.isfinite(gradient).all()
            or gradient.abs().max().item() == 0.0
        ):
            raise AssertionError("Invalid gradient for {}".format(name))

    block.eval()
    with torch.no_grad():
        block(**{name: value.detach() for name, value in inputs.items()})
    stats = block._last_stats
    assert stats["architecture_version"] == "fam_eppa_v4c"
    assert stats["adaptive_frequency_enabled"] is True
    assert stats["semantic_flow_enabled"] is True
    assert stats["haar_reconstruction_error"] < 1e-5
    assert abs(stats["alpf_kernel_sum"] - 1.0) < 1e-6
    assert abs(stats["ahpf_kernel_sum"] - 1.0) < 1e-6
    assert stats["flow_offset_max"] <= 1.5 + 1e-6
    assert stats["flow_offset_mean"] == 0.0
    assert stats["flow_alignment_delta_std"] < 2e-5
    for name, value in stats.items():
        if isinstance(value, float) and not torch.isfinite(torch.tensor(value)):
            raise AssertionError("Non-finite diagnostic: {}".format(name))


def main():
    check_config()
    check_identity_initialization()
    check_forward_backward_and_stats()
    print("FAM-EPPA V4-C checks passed.")


if __name__ == "__main__":
    main()

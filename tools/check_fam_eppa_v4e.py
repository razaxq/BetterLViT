"""Fast structural checks for FAM-EPPA V4-E PLAM calibration."""

from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import Config as config  # noqa: E402
from nets.eppa import FAMReliabilityCalibratedEPPA  # noqa: E402


def build_block(use_calibration):
    return FAMReliabilityCalibratedEPPA(
        64,
        text_dim=768,
        min_bottleneck_channels=32,
        use_adaptive_frequency=True,
        frequency_groups=8,
        use_semantic_flow_alignment=False,
        use_token_routing=False,
        use_plam_calibration=use_calibration,
        plam_calibration_max_delta=0.50,
        plam_calibration_hidden_channels=16,
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
    assert config.experiment_architecture_version == "fam_eppa_v4e"
    assert config.boundary_loss_weight == 0.0
    assert config.loss_name == "dice_focal"
    assert config.resume_path == ""
    assert config.require_checkpoint_architecture_match is True
    vit_config = config.get_CTranS_config()
    assert tuple(vit_config.eppa_adaptive_frequency_stages) == (
        "up4",
        "up3",
    )
    assert tuple(vit_config.eppa_semantic_flow_stages) == ()
    assert tuple(vit_config.eppa_token_routing_stages) == ()
    assert tuple(vit_config.eppa_plam_calibration_stages) == ("up4",)
    assert vit_config.eppa_plam_calibration_max_delta == 0.50


def check_identity_initialization():
    torch.manual_seed(1226)
    calibrated = build_block(use_calibration=True).eval()
    baseline = build_block(use_calibration=False).eval()
    incompatible = baseline.load_state_dict(
        calibrated.state_dict(),
        strict=False,
    )
    assert not incompatible.missing_keys
    assert all(
        "plam_calibrator" in name
        for name in incompatible.unexpected_keys
    )
    inputs = sample_inputs()
    with torch.no_grad():
        calibrated_output, calibrated_decoder = calibrated(
            **inputs,
            return_decoder=True,
        )
        baseline_output, baseline_decoder = baseline(
            **inputs,
            return_decoder=True,
        )
    if not torch.allclose(
        calibrated_decoder,
        baseline_decoder,
        atol=1e-6,
    ):
        raise AssertionError("V4-E must preserve the V4-B decoder path")
    if not torch.allclose(
        calibrated_output,
        baseline_output,
        atol=1e-6,
    ):
        raise AssertionError("Unit PLAM gate must reproduce V4-B")


def check_forward_backward_and_stats():
    torch.manual_seed(1227)
    block = build_block(use_calibration=True).train()
    # Production starts with a unit gate. Open the final projection only in
    # this reachability check so gradients can reach the hidden calibrator.
    torch.nn.init.normal_(block.plam_calibrator[-1].weight, std=0.02)
    inputs = sample_inputs()
    for name in ("skip", "plam", "decoder", "text"):
        inputs[name].requires_grad_(True)
    output, decoder = block(**inputs, return_decoder=True)
    assert output.shape == inputs["skip"].shape
    assert decoder.shape == inputs["decoder"].shape
    assert torch.isfinite(output).all() and torch.isfinite(decoder).all()
    (output.square().mean() + decoder.square().mean()).backward()

    parameters = dict(block.named_parameters())
    required_gradients = (
        "plam_calibrator.0.weight",
        "plam_calibrator.3.weight",
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
    assert stats["architecture_version"] == "fam_eppa_v4e"
    assert stats["adaptive_frequency_enabled"] is True
    assert stats["semantic_flow_enabled"] is False
    assert stats["token_routing_enabled"] is False
    assert stats["plam_calibration_enabled"] is True
    assert stats["haar_reconstruction_error"] < 1e-5
    assert abs(stats["alpf_kernel_sum"] - 1.0) < 1e-6
    assert abs(stats["ahpf_kernel_sum"] - 1.0) < 1e-6
    assert 0.5 <= stats["plam_calibration_gate_min"]
    assert stats["plam_calibration_gate_max"] <= 1.5
    assert stats["plam_calibration_gate_std"] > 0.0
    for name, value in stats.items():
        if isinstance(value, float) and not torch.isfinite(torch.tensor(value)):
            raise AssertionError("Non-finite diagnostic: {}".format(name))


def main():
    check_config()
    check_identity_initialization()
    check_forward_backward_and_stats()
    print("FAM-EPPA V4-E checks passed.")


if __name__ == "__main__":
    main()

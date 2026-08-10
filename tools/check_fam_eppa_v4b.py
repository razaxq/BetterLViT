"""Fast structural checks for the FAM-EPPA V4-B experiment."""

from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import Config as config  # noqa: E402
from nets.eppa import FAMAdaptiveHaarEPPA, FixedHaarFrequencySplit  # noqa: E402


def check_config():
    assert config.experiment_architecture_version == "fam_eppa_v4b"
    assert config.boundary_loss_weight == 0.0
    assert config.loss_name == "dice_focal"
    assert config.resume_path == ""
    assert config.require_checkpoint_architecture_match is True
    vit_config = config.get_CTranS_config()
    assert tuple(vit_config.eppa_adaptive_frequency_stages) == ("up4", "up3")
    assert vit_config.eppa_frequency_groups == 8
    assert 0.0 < vit_config.eppa_alpf_strength_init < 1.0
    assert 0.0 < vit_config.eppa_ahpf_strength_init < 1.0


def check_haar():
    torch.manual_seed(1221)
    for channels, height, width in ((3, 8, 10), (4, 7, 9), (64, 28, 28)):
        inputs = torch.randn(2, channels, height, width)
        low, high = FixedHaarFrequencySplit(channels)(inputs)
        error = (inputs - low - high).abs().amax().item()
        if error >= 1e-5:
            raise AssertionError(
                "Haar reconstruction failed for {}x{}x{}: {:.3e}".format(
                    channels,
                    height,
                    width,
                    error,
                )
            )


def check_adaptive_forward_backward():
    torch.manual_seed(1221)
    block = FAMAdaptiveHaarEPPA(
        64,
        text_dim=768,
        min_bottleneck_channels=32,
        use_adaptive_frequency=True,
        frequency_groups=8,
    )
    block.train()
    inputs = {
        "skip": torch.randn(2, 64, 32, 32, requires_grad=True),
        "plam": torch.randn(2, 64, 32, 32, requires_grad=True),
        "decoder": torch.randn(2, 64, 32, 32, requires_grad=True),
        "text": torch.randn(2, 32, 768, requires_grad=True),
    }
    output, decoder = block(**inputs, return_decoder=True)
    assert output.shape == inputs["skip"].shape
    assert decoder.shape == inputs["decoder"].shape
    assert torch.isfinite(output).all()
    assert torch.isfinite(decoder).all()
    (output.square().mean() + decoder.square().mean()).backward()

    parameters = dict(block.named_parameters())
    required_gradients = (
        "adaptive_frequency.low_kernel_predictor.weight",
        "adaptive_frequency.high_kernel_predictor.weight",
        "adaptive_frequency.alpf_strength_logit",
        "adaptive_frequency.ahpf_strength_logit",
        "region_out.3.weight",
        "detail_out.weight",
        "plam_strength_logit",
    )
    for name in required_gradients:
        gradient = parameters[name].grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise AssertionError("Missing or invalid gradient for {}".format(name))

    block.eval()
    with torch.no_grad():
        block(**{name: value.detach() for name, value in inputs.items()})
    stats = block._last_stats
    assert stats["architecture_version"] == "fam_eppa_v4b"
    assert stats["adaptive_frequency_enabled"] is True
    assert stats["haar_reconstruction_error"] < 1e-5
    assert abs(stats["alpf_kernel_sum"] - 1.0) < 1e-6
    assert abs(stats["ahpf_kernel_sum"] - 1.0) < 1e-6
    for prefix in ("alpf", "ahpf"):
        weight_sum = sum(
            stats["{}_{}_weight".format(prefix, name)]
            for name in ("identity", "blur3", "blur5")
        )
        assert abs(weight_sum - 1.0) < 1e-6
        assert 0.0 <= stats["{}_kernel_entropy".format(prefix)] <= 1.0


def check_nonadaptive_stage():
    block = FAMAdaptiveHaarEPPA(
        64,
        text_dim=768,
        min_bottleneck_channels=32,
        use_adaptive_frequency=False,
    )
    assert not hasattr(block, "adaptive_frequency")
    block.eval()
    inputs = torch.randn(1, 64, 16, 16)
    text = torch.randn(1, 32, 768)
    with torch.no_grad():
        output = block(inputs, inputs, inputs, text)
    assert output.shape == inputs.shape
    assert block._last_stats["adaptive_frequency_enabled"] is False


def main():
    check_config()
    check_haar()
    check_adaptive_forward_backward()
    check_nonadaptive_stage()
    print("FAM-EPPA V4-B checks passed.")


if __name__ == "__main__":
    main()

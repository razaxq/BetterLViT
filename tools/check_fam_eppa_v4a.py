"""Fast CPU checks for the FAM-EPPA V4-A structural experiment."""

from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import Config as config  # noqa: E402
from nets.eppa import FAMHaarEPPA, FixedHaarFrequencySplit  # noqa: E402


def check_config():
    assert config.experiment_architecture_version == "fam_eppa_v4a"
    assert config.boundary_loss_weight == 0.0
    assert config.loss_name == "dice_focal"
    assert config.resume_path == ""
    assert config.require_checkpoint_architecture_match is True
    vit_config = config.get_CTranS_config()
    assert vit_config.eppa_use_plam_guide is True
    assert vit_config.eppa_detail_strength_floor > 0.0


def check_haar():
    torch.manual_seed(1219)
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


def check_forward_backward():
    torch.manual_seed(1219)
    block = FAMHaarEPPA(
        64,
        text_dim=768,
        min_bottleneck_channels=32,
    )
    block.train()
    inputs = {
        "skip": torch.randn(2, 64, 32, 32, requires_grad=True),
        "plam": torch.randn(2, 64, 32, 32, requires_grad=True),
        "decoder": torch.randn(2, 64, 32, 32, requires_grad=True),
        "text": torch.randn(2, 32, 768, requires_grad=True),
    }
    output = block(**inputs)
    assert output.shape == inputs["skip"].shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()

    parameters = dict(block.named_parameters())
    required_gradients = (
        "region_out.3.weight",
        "detail_out.weight",
        "plam_strength_logit",
        "region_strength_logit",
        "detail_strength_logit",
    )
    for name in required_gradients:
        gradient = parameters[name].grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise AssertionError("Missing or invalid gradient for {}".format(name))

    block.eval()
    with torch.no_grad():
        block(**{name: value.detach() for name, value in inputs.items()})
    stats = block._last_stats
    assert stats["architecture_version"] == "fam_eppa_v4a"
    assert stats["haar_reconstruction_error"] < 1e-5
    assert 0.0 < stats["detail_strength_mean"] < 0.3
    assert abs(
        stats["skip_low_energy_ratio"]
        + stats["skip_high_energy_ratio"]
        - 1.0
    ) < 1e-5


def main():
    check_config()
    check_haar()
    check_forward_backward()
    print("FAM-EPPA V4-A checks passed.")


if __name__ == "__main__":
    main()

"""Fast structural checks for FAM-EPPA V4-D token routing."""

from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import Config as config  # noqa: E402
from nets.eppa import FAMTokenLocalizedEPPA  # noqa: E402


def build_block(use_tokens):
    return FAMTokenLocalizedEPPA(
        64,
        text_dim=768,
        min_bottleneck_channels=32,
        use_adaptive_frequency=True,
        frequency_groups=8,
        use_semantic_flow_alignment=False,
        use_token_routing=use_tokens,
        token_attention_dim=32,
        token_attention_heads=4,
        token_strength_init=0.10,
    )


def sample_inputs():
    text_mask = torch.tensor(
        [
            [1] * 11 + [0] * 21,
            [1] * 17 + [0] * 15,
        ],
        dtype=torch.long,
    )
    return {
        "skip": torch.randn(2, 64, 24, 24),
        "plam": torch.randn(2, 64, 24, 24),
        "decoder": torch.randn(2, 64, 24, 24),
        "text": torch.randn(2, 32, 768),
        "text_mask": text_mask,
    }


def check_config():
    assert config.experiment_architecture_version == "fam_eppa_v4d"
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
    assert tuple(vit_config.eppa_token_routing_stages) == (
        "up4",
        "up3",
    )
    assert vit_config.eppa_token_attention_dim == 32
    assert vit_config.eppa_token_attention_heads == 4


def check_identity_initialization():
    torch.manual_seed(1223)
    routed = build_block(use_tokens=True).eval()
    baseline = build_block(use_tokens=False).eval()
    incompatible = baseline.load_state_dict(
        routed.state_dict(),
        strict=False,
    )
    assert not incompatible.missing_keys
    assert all("token_" in name for name in incompatible.unexpected_keys)
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
    if not torch.allclose(routed_decoder, baseline_decoder, atol=1e-6):
        raise AssertionError("V4-D must preserve the V4-B decoder path")
    if not torch.allclose(routed_output, baseline_output, atol=1e-6):
        raise AssertionError("Zero token output must reproduce V4-B")


def check_padding_mask():
    torch.manual_seed(1224)
    block = build_block(use_tokens=True).eval()
    torch.nn.init.normal_(block.token_out.weight, std=0.02)
    inputs = sample_inputs()
    changed = dict(inputs)
    changed_text = inputs["text"].clone()
    changed_text[~inputs["text_mask"].bool()] = 1000.0
    changed["text"] = changed_text
    with torch.no_grad():
        first = block(**inputs)
        second = block(**changed)
    if not torch.allclose(first, second, atol=2e-5, rtol=1e-5):
        raise AssertionError("Padded text tokens changed the routed output")


def check_forward_backward_and_stats():
    torch.manual_seed(1225)
    block = build_block(use_tokens=True).train()
    # Both projections are intentionally zero-initialized so the production
    # model begins as V4-B. Open them only inside this gradient reachability
    # check; real training learns the outer region projection first.
    torch.nn.init.normal_(block.token_out.weight, std=0.02)
    torch.nn.init.normal_(block.region_out[-1].weight, std=0.02)
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
        "token_query.weight",
        "token_key.weight",
        "token_value.weight",
        "token_out.weight",
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
    assert stats["architecture_version"] == "fam_eppa_v4d"
    assert stats["adaptive_frequency_enabled"] is True
    assert stats["semantic_flow_enabled"] is False
    assert stats["token_routing_enabled"] is True
    assert stats["haar_reconstruction_error"] < 1e-5
    assert abs(stats["alpf_kernel_sum"] - 1.0) < 1e-6
    assert abs(stats["ahpf_kernel_sum"] - 1.0) < 1e-6
    assert 0.0 <= stats["token_attention_entropy"] <= 1.0 + 1e-6
    assert 0.0 < stats["token_non_cls_mass"] < 1.0
    assert stats["token_valid_count_mean"] == 14.0
    assert stats["token_residual_std"] > 0.0
    for name, value in stats.items():
        if isinstance(value, float) and not torch.isfinite(torch.tensor(value)):
            raise AssertionError("Non-finite diagnostic: {}".format(name))


def main():
    check_config()
    check_identity_initialization()
    check_padding_mask()
    check_forward_backward_and_stats()
    print("FAM-EPPA V4-D checks passed.")


if __name__ == "__main__":
    main()

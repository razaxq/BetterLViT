"""Deterministic CPU/GPU checks for Text-Conditioned Cross-Scale Skip Router."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nets.tcsr import (
    BoundaryPreservingAsymmetricTextGuidedRouter,
    CalibratedSingleHopBoundaryFocusedTextGuidedRouter,
    SparseBoundaryCalibratedTextGuidedRouter,
    SingleHopBoundaryFocusedTextGuidedRouter,
    SupervisedLocalSparseBoundaryTextGuidedRouter,
    TextConditionedCrossScaleSkipRouter,
    TextConditionedCrossScaleSkipRouterV2,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--version",
        choices=("v1", "v2", "v2.1", "v2.2", "v2.3", "v2.4", "v2.5"),
        default="v2",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1219)
    return parser.parse_args()


def require_nonzero_finite_gradient(name, parameter):
    gradient = parameter.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError("{} has no finite gradient.".format(name))
    if gradient.abs().sum().item() == 0.0:
        raise RuntimeError("{} has an all-zero gradient.".format(name))


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device("cuda" if args.cuda else "cpu")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False

    channels = (8, 16, 32, 64)
    spatial_sizes = (32, 16, 8, 4)
    if args.version in ("v2.3", "v2.4", "v2.5"):
        router_types = {
            "v2.3": CalibratedSingleHopBoundaryFocusedTextGuidedRouter,
            "v2.4": SparseBoundaryCalibratedTextGuidedRouter,
            "v2.5": SupervisedLocalSparseBoundaryTextGuidedRouter,
        }
        router = router_types[args.version](
            channels,
            text_dim=16,
            routing_dim=8,
            max_residual_strength=0.08,
            initial_residual_strength=0.04,
            initial_gate_probability=0.25,
            gate_min_probability=0.05,
            gate_max_probability=0.50,
            gate_target_min=0.15,
            gate_target_max=0.35,
            gate_calibration_weight=0.01,
            **({
                "localization_weight": 0.02,
                "residual_leakage_weight": 0.5,
            } if args.version == "v2.5" else {}),
        ).to(device)
    elif args.version == "v2.2":
        router = SingleHopBoundaryFocusedTextGuidedRouter(
            channels,
            text_dim=16,
            routing_dim=8,
            max_residual_strength=0.08,
            initial_residual_strength=0.04,
            initial_gate_probability=0.25,
        ).to(device)
    elif args.version == "v2.1":
        router = BoundaryPreservingAsymmetricTextGuidedRouter(
            channels,
            text_dim=16,
            routing_dim=8,
            max_residual_strength=0.15,
            initial_residual_strength=0.08,
            initial_gate_probability=0.15,
            gate_activation_budget=0.35,
            gate_budget_weight=0.02,
            gate_binary_weight=0.005,
        ).to(device)
    elif args.version == "v2":
        router = TextConditionedCrossScaleSkipRouterV2(
            channels,
            text_dim=16,
            routing_dim=8,
            max_residual_strength=0.5,
            initial_residual_strength=0.05,
        ).to(device)
    else:
        router = TextConditionedCrossScaleSkipRouter(
            channels,
            text_dim=16,
            routing_dim=8,
            max_residual_strength=1.0,
        ).to(device)
    router.train(args.version == "v2.5")
    skips = tuple(
        torch.randn(
            args.batch_size,
            channel,
            size,
            size,
            device=device,
            requires_grad=True,
        )
        for channel, size in zip(channels, spatial_sizes)
    )
    text = torch.randn(
        args.batch_size,
        7,
        16,
        device=device,
        requires_grad=True,
    )
    text_mask = torch.ones(
        args.batch_size,
        7,
        dtype=torch.long,
        device=device,
    )
    text_mask[0, -2:] = 0
    if args.batch_size > 1:
        text_mask[1].zero_()

    inference_equivalence_error = 0.0
    if args.version == "v2.5":
        reference = SparseBoundaryCalibratedTextGuidedRouter(
            channels,
            text_dim=16,
            routing_dim=8,
            max_residual_strength=0.08,
            initial_residual_strength=0.04,
            initial_gate_probability=0.25,
            gate_min_probability=0.05,
            gate_max_probability=0.50,
            gate_target_min=0.15,
            gate_target_max=0.35,
            gate_calibration_weight=0.01,
        ).to(device)
        reference.load_state_dict(router.state_dict(), strict=True)
        reference.eval()
        router.eval()
        with torch.no_grad():
            reference_outputs = reference(
                tuple(source.detach() for source in skips),
                text.detach(),
                text_mask=text_mask,
            )
            candidate_outputs = router(
                tuple(source.detach() for source in skips),
                text.detach(),
                text_mask=text_mask,
            )
        inference_equivalence_error = max(
            (left - right).abs().max().item()
            for left, right in zip(reference_outputs, candidate_outputs)
        )
        if inference_equivalence_error != 0.0:
            raise RuntimeError(
                "V2.5 changed the V2.4 inference path: {:.3e}."
                .format(inference_equivalence_error)
            )
        router.train()

    initial_outputs = router(skips, text, text_mask=text_mask)
    initial_error = max(
        (output - source).abs().max().item()
        for output, source in zip(initial_outputs, skips)
    )
    initial_rms_ratios = [
        float(
            ((output - source).square().mean().sqrt()
             / source.square().mean().sqrt().clamp_min(1e-8)).item()
        )
        for output, source in zip(initial_outputs, skips)
    ]
    if args.version == "v1" and initial_error != 0.0:
        raise RuntimeError(
            "V1 identity initialization changed a skip: {:.3e}.".format(
                initial_error
            )
        )
    if args.version in (
        "v2", "v2.1", "v2.2", "v2.3", "v2.4", "v2.5",
    ) and max(initial_rms_ratios) >= 0.08:
        raise RuntimeError(
            "V2 initial residual exceeded the 8% RMS safety bound: {}."
            .format(initial_rms_ratios)
        )
    if any(tuple(output.shape) != tuple(source.shape)
           for output, source in zip(initial_outputs, skips)):
        raise RuntimeError("TCSR changed one or more skip shapes.")
    if not all(torch.isfinite(output).all() for output in initial_outputs):
        raise RuntimeError("Initial pass produced a non-finite output.")
    first_stats = dict(router._last_stats)
    if (
        args.version == "v1"
        and abs(first_stats["scale_weight_sum"] - 1.0) > 1e-6
    ):
        raise RuntimeError("Scale routing weights do not sum to one.")

    if args.version == "v1":
        identity_loss = sum(output.square().mean() for output in initial_outputs)
        identity_loss.backward()
        require_nonzero_finite_gradient("residual_gate", router.residual_gate)
        router.zero_grad(set_to_none=True)
        for source in skips:
            source.grad = None
        text.grad = None
        with torch.no_grad():
            router.residual_gate.fill_(0.25)
    active_outputs_1 = router(skips, text, text_mask=text_mask)
    active_outputs_2 = router(skips, text, text_mask=text_mask)
    deterministic_error = max(
        (left - right).abs().max().item()
        for left, right in zip(active_outputs_1, active_outputs_2)
    )
    if deterministic_error != 0.0:
        raise RuntimeError(
            "Repeated deterministic forwards differ: {:.3e}.".format(
                deterministic_error
            )
        )
    active_loss = sum(output.square().mean() for output in active_outputs_1)
    localization_loss = active_loss.new_zeros(())
    localization_components = {}
    if args.version == "v2.5":
        labels = torch.zeros(
            args.batch_size,
            1,
            spatial_sizes[0],
            spatial_sizes[0],
            device=device,
        )
        labels[:, :, 8:24, 9:23] = 1.0
        localization_loss = router.supervised_localization_loss(
            labels,
            weight_scale=1.0,
        )
        if not torch.isfinite(localization_loss) or localization_loss <= 0.0:
            raise RuntimeError("V2.5 localization loss is not positive/finite.")
        localization_components = router.localization_components()
        active_loss = active_loss + localization_loss
    active_loss.backward()
    if args.version == "v2":
        gradient_checks = {
            "residual_strength_logit": router.residual_strength_logit,
            "visual_projection": router.visual_projections[0][0].weight,
            "consensus_fusion": router.consensus_fusions[0][0].weight,
            "text_key": router.text_key.weight,
            "route_confidence": router.route_confidence_heads[0][-1].weight,
            "film_head": router.film_heads[0].weight,
            "channel_head": router.channel_heads[0].weight,
            "spatial_head": router.spatial_heads[0].weight,
            "message_head": router.message_heads[0][0].weight,
        }
    elif args.version == "v2.1":
        gradient_checks = {
            "residual_strength_logit": router.residual_strength_logit,
            "source_projection": router.visual_projections[3][0].weight,
            "target_projection": router.visual_projections[2][0].weight,
            "route_fusion": router.route_fusions[0][0].weight,
            "text_key": router.text_key.weight,
            "abstention_head": router.abstention_heads[0][-1].weight,
            "film_head": router.film_heads[0].weight,
            "channel_head": router.channel_heads[0].weight,
            "spatial_head": router.spatial_heads[0].weight,
            "message_head": router.message_heads[0].weight,
        }
    elif args.version in ("v2.2", "v2.3", "v2.4", "v2.5"):
        gradient_checks = {
            "residual_strength_logit": router.residual_strength_logit,
            "source_projection": router.source_projection[0].weight,
            "target_projection": router.target_projection[0].weight,
            "route_fusion": router.route_fusion[0].weight,
            "text_key": router.text_key.weight,
            "route_confidence": router.route_confidence[-1].weight,
            "film_head": router.film_head.weight,
            "channel_head": router.channel_head.weight,
            "spatial_head": router.spatial_head.weight,
            "message_head": router.message_head.weight,
        }
    else:
        gradient_checks = {
            "residual_gate": router.residual_gate,
            "visual_projection": router.visual_projections[0][0].weight,
            "text_key": router.text_key.weight,
            "scale_score": router.scale_score_heads[0][-1].weight,
            "channel_head": router.channel_heads[0].weight,
            "spatial_head": router.spatial_local[0].weight,
        }
    for name, parameter in gradient_checks.items():
        require_nonzero_finite_gradient(name, parameter)

    active_stats = dict(router._last_stats)
    with torch.no_grad():
        shifted_text = text.detach().clone()
        shifted_text[:, 0] = shifted_text[:, 0] + 0.25
        text_shift_outputs = router(
            tuple(source.detach() for source in skips),
            shifted_text,
            text_mask=text_mask,
        )
        text_conditioning_effect = max(
            (left.detach() - right).abs().max().item()
            for left, right in zip(active_outputs_1, text_shift_outputs)
        )
        changed_skips = [source.detach().clone() for source in skips]
        if args.version == "v2.1":
            changed_source_index, changed_target_index = 3, 2
        elif args.version in ("v2.2", "v2.3", "v2.4", "v2.5"):
            changed_source_index, changed_target_index = 2, 1
        else:
            changed_source_index, changed_target_index = 0, 1
        changed_skips[changed_source_index] = (
            changed_skips[changed_source_index] + 0.25
        )
        cross_scale_outputs = router(
            tuple(changed_skips),
            text.detach(),
            text_mask=text_mask,
        )
        adjacent_scale_effect = (
            active_outputs_1[changed_target_index].detach()
            - cross_scale_outputs[changed_target_index]
        ).abs().max().item()
    if args.version in (
        "v2", "v2.1", "v2.2", "v2.3", "v2.4", "v2.5",
    ) and text_conditioning_effect == 0.0:
        raise RuntimeError("V2 output is insensitive to text conditioning.")
    if args.version in (
        "v2", "v2.1", "v2.2", "v2.3", "v2.4", "v2.5",
    ) and adjacent_scale_effect == 0.0:
        raise RuntimeError("V2 does not exchange adjacent-scale features.")
    if args.version == "v2.1":
        identity_error = max(
            (active_outputs_1[index] - skips[index]).abs().max().item()
            for index in (0, 3)
        )
        if identity_error != 0.0:
            raise RuntimeError(
                "V2.1 changed boundary-preserving identity scales: {:.3e}."
                .format(identity_error)
            )
        regularization = router.regularization_loss()
        if not torch.isfinite(regularization):
            raise RuntimeError("V2.1 regularization is non-finite.")
    elif args.version in ("v2.2", "v2.3", "v2.4", "v2.5"):
        identity_error = max(
            (active_outputs_1[index] - skips[index]).abs().max().item()
            for index in (0, 2, 3)
        )
        if identity_error != 0.0:
            raise RuntimeError(
                "V2.2 changed protected identity scales: {:.3e}.".format(
                    identity_error
                )
            )
        regularization = router.regularization_loss()
        if not torch.isfinite(regularization):
            raise RuntimeError("Single-hop gate regularization is non-finite.")
        if args.version == "v2.2" and regularization.item() != 0.0:
            raise RuntimeError("V2.2 must not apply gate regularization.")
        if args.version in ("v2.3", "v2.4", "v2.5"):
            gate_mean = active_stats["route_gate_means"][0]
            if not 0.05 <= gate_mean <= 0.50:
                raise RuntimeError("V2.3 gate escaped its calibrated bounds.")
        if args.version in ("v2.4", "v2.5"):
            focus_mean = active_stats["boundary_focus_means"][0]
            if not 0.0 < focus_mean < 0.50:
                raise RuntimeError(
                    "V2.4 sparse boundary focus is not selective: {:.4f}."
                    .format(focus_mean)
                )

    result = {
        "status": "ok",
        "architecture_version": router.architecture_version,
        "device": str(device),
        "batch_size": args.batch_size,
        "parameter_count": sum(
            parameter.numel() for parameter in router.parameters()
        ),
        "initial_max_abs_delta": initial_error,
        "initial_delta_rms_ratios": initial_rms_ratios,
        "deterministic_repeat_max_abs_error": deterministic_error,
        "text_conditioning_max_abs_effect": text_conditioning_effect,
        "adjacent_scale_max_abs_effect": adjacent_scale_effect,
        "all_required_gradients_nonzero": True,
        "v2_4_inference_equivalence_max_abs_error": (
            inference_equivalence_error
        ),
    }
    if args.version == "v2":
        result.update({
            "route_confidences": active_stats["route_confidences"],
            "effective_strengths": active_stats["effective_strengths"],
            "delta_rms_ratios": active_stats["delta_rms_ratios"],
        })
    elif args.version == "v2.1":
        result.update({
            "route_names": active_stats["route_names"],
            "route_gate_means": active_stats["route_gate_means"],
            "route_gate_closed_fractions": active_stats[
                "route_gate_closed_fractions"
            ],
            "effective_strengths": active_stats["effective_strengths"],
            "delta_rms_ratios": active_stats["delta_rms_ratios"],
            "identity_scales": active_stats["identity_scales"],
            "regularization_loss": active_stats["regularization_loss"],
        })
    elif args.version in ("v2.2", "v2.3", "v2.4", "v2.5"):
        result.update({
            "route_names": active_stats["route_names"],
            "route_gate_means": active_stats["route_gate_means"],
            "route_gate_closed_fractions": active_stats[
                "route_gate_closed_fractions"
            ],
            "boundary_focus_means": active_stats["boundary_focus_means"],
            "effective_strengths": active_stats["effective_strengths"],
            "delta_rms_ratios": active_stats["delta_rms_ratios"],
            "identity_scales": active_stats["identity_scales"],
            "regularization_loss": active_stats["regularization_loss"],
        })
        if args.version in ("v2.3", "v2.4", "v2.5"):
            result.update({
                "gate_min_probability": active_stats["gate_min_probability"],
                "gate_max_probability": active_stats["gate_max_probability"],
                "gate_target_min": active_stats["gate_target_min"],
                "gate_target_max": active_stats["gate_target_max"],
                "gate_calibration_penalty": active_stats[
                    "gate_calibration_penalty"
                ],
            })
        if args.version == "v2.5":
            result.update({
                "localization_loss": float(localization_loss.item()),
                "localization_components": {
                    name: float(value.item())
                    for name, value in localization_components.items()
                },
            })
    else:
        result.update({
            "scale_weights": active_stats["scale_weights"],
            "scale_weight_sum": active_stats["scale_weight_sum"],
            "effective_gates": active_stats["effective_gates"],
        })
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

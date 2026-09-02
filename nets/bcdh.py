# -*- coding: utf-8 -*-
"""Boundary-conscious dual-head output refinement without boundary targets."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCDHRefiner(nn.Module):
    """Refine fine logits using prediction-only cross-scale cues.

    Both heads predict the complete segmentation mask.  The refinement cues are
    detached so neither prediction head can game the cue construction.  The
    final projection is zero-initialized, making the module an exact identity
    with respect to the fine/base logits at initialization.
    """

    architecture_version = "bcdh_r_v1"

    def __init__(
        self,
        channels=64,
        hidden_channels=32,
        delta_max=1.0,
        detach_cues=True,
    ):
        super().__init__()
        if channels <= 0 or hidden_channels <= 0:
            raise ValueError("BCDH channels must be positive")
        if hidden_channels % 8 != 0:
            raise ValueError("BCDH hidden channels must be divisible by 8")
        if delta_max <= 0.0:
            raise ValueError("BCDH delta_max must be positive")

        self.delta_max = float(delta_max)
        self.detach_cues = bool(detach_cues)
        self.coarse_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.refiner = nn.Sequential(
            nn.Conv2d(
                channels + 4,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=hidden_channels,
                bias=False,
            ),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        nn.init.zeros_(self.refiner[-1].weight)
        nn.init.zeros_(self.refiner[-1].bias)
        self._last_stats = {}

    def _cue(self, tensor):
        return tensor.detach() if self.detach_cues else tensor

    def _record_stats(
        self,
        coarse_probability,
        base_probability,
        uncertainty,
        fine_only,
        coarse_only,
        delta,
    ):
        if self.training:
            return
        with torch.no_grad():
            absolute_delta = delta.abs().flatten()
            uncertainty_flat = uncertainty.flatten()
            count = max(1, absolute_delta.numel() // 5)
            top_indices = torch.topk(
                uncertainty_flat,
                k=count,
                largest=True,
                sorted=False,
            ).indices
            top_energy = absolute_delta[top_indices].mean()
            all_energy = absolute_delta.mean()
            rest_count = absolute_delta.numel() - count
            if rest_count > 0:
                rest_energy = (
                    absolute_delta.sum()
                    - absolute_delta[top_indices].sum()
                ) / rest_count
            else:
                rest_energy = absolute_delta.new_zeros(())
            self._last_stats = {
                "architecture_version": self.architecture_version,
                "coarse_probability_mean": float(
                    coarse_probability.mean().item()
                ),
                "base_probability_mean": float(
                    base_probability.mean().item()
                ),
                "uncertainty_mean": float(uncertainty.mean().item()),
                "fine_only_mean": float(fine_only.mean().item()),
                "coarse_only_mean": float(coarse_only.mean().item()),
                "delta_abs_mean": float(all_energy.item()),
                "delta_abs_max": float(absolute_delta.max().item()),
                "delta_positive_fraction": float((delta > 0).float().mean().item()),
                "delta_negative_fraction": float((delta < 0).float().mean().item()),
                "uncertainty_top20_delta_abs_mean": float(top_energy.item()),
                "uncertainty_rest_delta_abs_mean": float(rest_energy.item()),
            }

    def forward(self, coarse_feature, fine_feature, base_logits):
        if coarse_feature.shape[0] != fine_feature.shape[0]:
            raise ValueError("BCDH coarse/fine batch sizes differ")
        if fine_feature.shape[-2:] != base_logits.shape[-2:]:
            raise ValueError("BCDH fine feature and base logit sizes differ")

        coarse_logits = self.coarse_head(coarse_feature)
        coarse_logits = F.interpolate(
            coarse_logits,
            size=base_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        coarse_probability = torch.sigmoid(coarse_logits)
        base_probability = torch.sigmoid(base_logits)

        coarse_cue = self._cue(coarse_probability)
        base_cue = self._cue(base_probability)
        uncertainty = 4.0 * coarse_cue * (1.0 - coarse_cue)
        fine_only = base_cue * (1.0 - coarse_cue)
        coarse_only = coarse_cue * (1.0 - base_cue)
        refiner_input = torch.cat(
            [
                fine_feature,
                coarse_cue,
                uncertainty,
                fine_only,
                coarse_only,
            ],
            dim=1,
        )
        delta = self.delta_max * torch.tanh(self.refiner(refiner_input))
        final_logits = base_logits + delta
        final_probability = torch.sigmoid(final_logits)

        self._record_stats(
            coarse_probability,
            base_probability,
            uncertainty,
            fine_only,
            coarse_only,
            delta,
        )
        return {
            "final": final_probability,
            "coarse": coarse_probability,
            "base": base_probability,
            "delta": delta,
        }

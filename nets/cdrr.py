# -*- coding: utf-8 -*-
"""Cross-scale detail reliability refinement without boundary supervision."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CDRRRefiner(nn.Module):
    """Apply balanced corrections only at reliable cross-scale detail sites.

    The auxiliary coarse head and all refiner inputs are detached from the
    segmentation trunk.  Consequently, auxiliary supervision cannot distort
    the parent decoder.  A deterministic top-k support prevents the residual
    from becoming a global logit bias, while support-weighted centering keeps
    both correction directions available.
    """

    architecture_version = "cdrr_v1"

    def __init__(
        self,
        channels=64,
        hidden_channels=32,
        delta_max=0.5,
        active_fraction=0.15,
    ):
        super().__init__()
        if channels <= 0 or hidden_channels <= 0:
            raise ValueError("CDRR channels must be positive")
        if hidden_channels % 8 != 0:
            raise ValueError("CDRR hidden channels must be divisible by 8")
        if delta_max <= 0.0:
            raise ValueError("CDRR delta_max must be positive")
        if not 0.0 < active_fraction < 0.5:
            raise ValueError("CDRR active_fraction must be in (0, 0.5)")

        self.delta_max = float(delta_max)
        self.active_fraction = float(active_fraction)
        self.coarse_head = nn.Conv2d(channels, 1, kernel_size=1)
        cue_channels = 8
        self.refiner = nn.Sequential(
            nn.Conv2d(
                channels + cue_channels,
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

    @staticmethod
    def _detail(feature):
        local_mean = F.avg_pool2d(
            feature,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        magnitude = (feature - local_mean).abs().mean(dim=1, keepdim=True)
        scale = magnitude.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (magnitude / (4.0 * scale)).clamp(0.0, 1.0)

    def _support(self, reliability):
        flat = reliability.flatten(1)
        count = max(1, int(round(flat.shape[1] * self.active_fraction)))
        indices = torch.topk(
            flat,
            k=count,
            dim=1,
            largest=True,
            sorted=False,
        ).indices
        support = torch.zeros_like(flat)
        support.scatter_(1, indices, 1.0)
        return support.view_as(reliability)

    def _record_stats(
        self,
        reliability,
        support,
        agreement,
        uncertainty,
        disagreement,
        delta,
    ):
        if self.training:
            return
        with torch.no_grad():
            active = support > 0
            active_delta = delta[active]
            inactive_abs_max = delta[~active].abs().max()
            self._last_stats = {
                "architecture_version": self.architecture_version,
                "support_fraction": float(support.mean().item()),
                "reliability_mean": float(reliability.mean().item()),
                "agreement_mean": float(agreement.mean().item()),
                "uncertainty_mean": float(uncertainty.mean().item()),
                "disagreement_mean": float(disagreement.mean().item()),
                "delta_mean": float(delta.mean().item()),
                "delta_abs_mean": float(delta.abs().mean().item()),
                "delta_abs_active_mean": float(active_delta.abs().mean().item()),
                "delta_abs_max": float(delta.abs().max().item()),
                "delta_positive_active_fraction": float(
                    (active_delta > 0).float().mean().item()
                ),
                "delta_negative_active_fraction": float(
                    (active_delta < 0).float().mean().item()
                ),
                "delta_inactive_abs_max": float(inactive_abs_max.item()),
            }

    def forward(self, coarse_feature, fine_feature, base_logits):
        if coarse_feature.shape[0] != fine_feature.shape[0]:
            raise ValueError("CDRR coarse/fine batch sizes differ")
        if fine_feature.shape[-2:] != base_logits.shape[-2:]:
            raise ValueError("CDRR fine feature and base logit sizes differ")

        detached_coarse = coarse_feature.detach()
        detached_fine = fine_feature.detach()
        coarse_logits = self.coarse_head(detached_coarse)
        coarse_logits = F.interpolate(
            coarse_logits,
            size=base_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        coarse_probability = torch.sigmoid(coarse_logits)
        base_probability = torch.sigmoid(base_logits)

        coarse_cue = coarse_probability.detach()
        base_cue = base_probability.detach()
        uncertainty = 4.0 * base_cue * (1.0 - base_cue)
        disagreement = (base_cue - coarse_cue).abs()
        fine_only = base_cue * (1.0 - coarse_cue)
        coarse_only = coarse_cue * (1.0 - base_cue)

        coarse_up = F.interpolate(
            detached_coarse,
            size=fine_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fine_detail = self._detail(detached_fine)
        coarse_detail = self._detail(coarse_up)
        detail_agreement = torch.exp(
            -2.0 * (fine_detail - coarse_detail).abs()
        )
        shared_detail = torch.sqrt(
            (fine_detail * coarse_detail).clamp_min(0.0)
        )
        reliability = detail_agreement * (
            0.45 * uncertainty
            + 0.35 * disagreement
            + 0.20 * shared_detail
        )
        support = self._support(reliability.detach())

        refiner_input = torch.cat(
            [
                detached_fine,
                coarse_cue,
                base_cue,
                uncertainty,
                disagreement,
                fine_only,
                coarse_only,
                detail_agreement,
                shared_detail,
            ],
            dim=1,
        )
        raw_score = self.refiner(refiner_input)
        support_count = support.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        support_mean = (raw_score * support).sum(
            dim=(2, 3), keepdim=True
        ) / support_count
        centered_score = raw_score - support_mean
        delta = (
            support
            * self.delta_max
            * torch.tanh(centered_score)
        )
        final_probability = torch.sigmoid(base_logits + delta)

        self._record_stats(
            reliability,
            support,
            detail_agreement,
            uncertainty,
            disagreement,
            delta,
        )
        return {
            "final": final_probability,
            "coarse": coarse_probability,
            "base": base_probability,
            "delta": delta,
            "support": support,
            "reliability": reliability,
            "fine_detail": fine_detail,
            "coarse_detail": coarse_detail,
        }

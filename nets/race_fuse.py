"""Report-Anatomy Consistency and Evidence Fusion (RACE-Fuse V1).

RACE predicts compact report slots from frozen CXR-BERT features, compares
them with visual evidence at every encoder scale, and opens only positive,
anatomically aligned residual routes.  Each route is exact identity at
initialization, so the control network is recovered before learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _RACERoute(nn.Module):
    def __init__(self, channels, hidden_channels, max_strength):
        super().__init__()
        hidden = min(int(hidden_channels), int(channels))
        self.evidence = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                      groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.GELU(),
        )
        self.strength_logit = nn.Parameter(torch.zeros(()))
        self.max_strength = float(max_strength)

    def forward(self, skip, report_prior, zone_basis, text_zones):
        basis = F.interpolate(
            zone_basis.float(), size=skip.shape[-2:], mode="nearest"
        )
        prior = F.interpolate(
            report_prior.float(), size=skip.shape[-2:], mode="nearest"
        )
        evidence_logits = self.evidence(skip)
        evidence = torch.sigmoid(evidence_logits)
        denominators = basis.sum(dim=(2, 3)).clamp_min(1.0)
        zone_evidence = (
            evidence * basis
        ).sum(dim=(2, 3)) / denominators
        mentioned = text_zones.detach()
        agreement = 1.0 - (
            (zone_evidence - text_zones).abs() * mentioned
        ).sum(dim=1, keepdim=True) / mentioned.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        agreement = agreement.clamp(0.0, 1.0).view(-1, 1, 1, 1)
        gate = prior * evidence * agreement
        strength = self.max_strength * torch.tanh(self.strength_logit)
        routed = skip + strength * gate * self.residual(skip)
        stats = {
            "strength": float(strength.detach().cpu()),
            "gate_mean": float(gate.detach().mean().cpu()),
            "evidence_mean": float(evidence.detach().mean().cpu()),
            "agreement_mean": float(agreement.detach().mean().cpu()),
        }
        return routed, zone_evidence, stats


class RACEFuse(nn.Module):
    architecture_version = "race_fuse_v1"

    def __init__(
        self,
        channels=(64, 128, 256, 512),
        text_dim=768,
        hidden_channels=32,
        max_strength=0.15,
    ):
        super().__init__()
        self.slot_head = nn.Sequential(
            nn.Linear(text_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 9),
        )
        self.routes = nn.ModuleList([
            _RACERoute(value, hidden_channels, max_strength)
            for value in channels
        ])
        self._last_stats = {}

    @staticmethod
    def _masked_mean(text, text_mask):
        if text_mask is None:
            return text.mean(dim=1)
        weights = text_mask.to(dtype=text.dtype).unsqueeze(-1)
        return (text * weights).sum(dim=1) / weights.sum(
            dim=1
        ).clamp_min(1.0)

    def forward(self, skips, text, text_mask, zone_basis):
        if zone_basis is None:
            raise ValueError("RACE-Fuse requires the transformed zone basis")
        slot_logits = self.slot_head(self._masked_mean(text, text_mask))
        slot_probabilities = torch.sigmoid(slot_logits)
        text_zones = slot_probabilities[:, :6]
        report_prior = (
            text_zones[:, :, None, None] * zone_basis.float()
        ).sum(dim=1, keepdim=True).clamp(0.0, 1.0)

        routed = []
        visual_zones = []
        route_stats = []
        for route, skip in zip(self.routes, skips):
            value, zone_evidence, stats = route(
                skip, report_prior, zone_basis, text_zones
            )
            routed.append(value)
            visual_zones.append(zone_evidence)
            route_stats.append(stats)
        self._last_stats = {
            "architecture_version": self.architecture_version,
            "slot_probability_mean": float(
                slot_probabilities.detach().mean().cpu()
            ),
            "route_strengths": [row["strength"] for row in route_stats],
            "route_gate_means": [row["gate_mean"] for row in route_stats],
            "route_evidence_means": [
                row["evidence_mean"] for row in route_stats
            ],
            "route_agreement_means": [
                row["agreement_mean"] for row in route_stats
            ],
        }
        return tuple(routed), {
            "slot_logits": slot_logits,
            "visual_zone_probabilities": visual_zones,
            "report_prior": report_prior,
        }

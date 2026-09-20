"""RACE-PE: separate regional presence from pixel extent evidence."""
import torch
from torch import nn
from torch.nn import functional as F

from .race_fuse import RACEFuse, _RACERoute


class _PERoute(_RACERoute):
    def __init__(self, channels, hidden_channels, max_strength):
        super().__init__(channels, hidden_channels, max_strength)
        self.presence = nn.Linear(channels, 1)

    def forward(self, skip, prior, zone_basis, text_zones):
        basis = F.interpolate(zone_basis.float(), skip.shape[-2:], mode="nearest")
        mass = basis.sum((2, 3))
        pooled = torch.einsum("bchw,bzhw->bzc", skip, basis)
        pooled = pooled / mass.clamp_min(1).unsqueeze(-1)
        presence_logits = self.presence(pooled).squeeze(-1)
        presence = presence_logits.sigmoid()
        extent_logits = self.evidence(skip)
        extent = extent_logits.sigmoid()
        occupancy = (extent * basis).sum((2, 3)) / mass.clamp_min(1)
        # Presence compares with presence; occupancy never compares with text.
        agreement_zones = 1 - (presence - text_zones).abs()
        support = (text_zones * agreement_zones * (mass > 0)).unsqueeze(-1).unsqueeze(-1)
        gate = (support * basis).sum(1, keepdim=True) * extent
        strength = self.max_strength * self.strength_logit.tanh()
        routed = skip + strength * gate * self.residual(skip)
        return routed, {
            "extent_logits": extent_logits,
            "presence_logits": presence_logits,
            "occupancy": occupancy,
            "basis": basis,
        }, {
            "strength": float(strength.detach()),
            "gate_mean": float(gate.detach().mean()),
            "evidence_mean": float(extent.detach().mean()),
            "agreement_mean": float(agreement_zones.detach().mean()),
            "presence_mean": float(presence.detach().mean()),
            "occupancy_mean": float(occupancy.detach().mean()),
        }


class RACEPE(RACEFuse):
    architecture_version = "race_pe_v1"

    def __init__(self, channels=(64, 128, 256, 512), text_dim=768,
                 hidden_channels=32, max_strength=0.15, route_enabled=True):
        super().__init__(channels, text_dim, hidden_channels, max_strength)
        self.routes = nn.ModuleList([
            _PERoute(c, hidden_channels, max_strength) for c in channels
        ])
        self.route_enabled = route_enabled

    def forward(self, skips, text, text_mask, zone_basis):
        if zone_basis is None:
            raise ValueError("RACE-PE requires transformed zone bases")
        slots = self.slot_head(self._masked_mean(text, text_mask))
        zones = slots[:, :6].sigmoid()
        routed, routes, stats = [], [], []
        for module, skip in zip(self.routes, skips):
            value, auxiliary, row = module(skip, None, zone_basis, zones)
            routed.append(value if self.route_enabled else skip)
            routes.append(auxiliary)
            stats.append(row)
        self._last_stats = {
            "architecture_version": self.architecture_version,
            "slot_probability_mean": float(zones.detach().mean()),
            **{"route_" + key: [row[source] for row in stats] for key, source in (
                ("strengths", "strength"), ("gate_means", "gate_mean"),
                ("evidence_means", "evidence_mean"),
                ("agreement_means", "agreement_mean"),
                ("presence_means", "presence_mean"),
                ("occupancy_means", "occupancy_mean"),
            )},
        }
        return tuple(routed), {"slot_logits": slots, "pe_routes": routes}

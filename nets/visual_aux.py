"""C8 visual heads as training-only supervision; never modify a skip tensor."""
import torch
from torch.nn import functional as F
from .race_pe import RACEPE


class VisualAuxiliaryHeads(RACEPE):
    architecture_version = 'r2_visual_aux_v1'

    def __init__(self, mode, **kwargs):
        if mode not in ('pixel', 'regional'):
            raise ValueError('Expected a registered visual auxiliary mode')
        super().__init__(route_enabled=False, **kwargs)
        self.mode = mode
        self.slot_head.requires_grad_(False)
        for route in self.routes:
            route.residual.requires_grad_(False)
            route.strength_logit.requires_grad_(False)
            if mode == 'pixel': route.presence.requires_grad_(False)

    def forward(self, skips, text, text_mask, zone_basis):
        if zone_basis is None: raise ValueError('Transformed six-region basis required')
        rows=[]
        for route, skip in zip(self.routes, skips):
            basis=F.interpolate(zone_basis.float(),skip.shape[-2:],mode='nearest')
            mass=basis.sum((2,3))
            extent_logits=route.evidence(skip)
            occupancy=(extent_logits.sigmoid()*basis).sum((2,3))/mass.clamp_min(1)
            pooled=torch.einsum('bchw,bzhw->bzc',skip,basis)/mass.clamp_min(1).unsqueeze(-1)
            presence_logits=route.presence(pooled).squeeze(-1)
            rows.append(dict(extent_logits=extent_logits,presence_logits=presence_logits,
                             occupancy=occupancy,basis=basis))
        self._last_stats={}
        return tuple(skips), {'pe_routes':rows}

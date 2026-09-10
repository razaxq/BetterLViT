"""Capacity-matched existing-evidence / raw-image redecoders; no novelty claim."""
import torch
from torch import nn
from torch.nn import functional as F


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.GroupNorm(8, cout), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                         nn.GroupNorm(8, cout), nn.GELU())


def region_mask(probability):
    """Stable ties by flat index; accepts predictions only, never targets."""
    scores = F.avg_pool2d(4 * probability * (1 - probability), 16).flatten(1)
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    chosen = torch.zeros_like(scores).scatter(1, order[:, :40], 1)
    return F.interpolate(chosen.reshape(-1, 1, 14, 14), scale_factor=16, mode='nearest')


class Reencoder(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.stem = block(3, 16)
        self.image_down = block(16, 32)
        self.feature_project = nn.Conv2d(320, 32, 1)
        self.fuse = block(66, 48)
        self.context = block(48, 64)
        self.up_half = block(112, 32)
        self.up_full = block(48, 16)
        self.out = nn.Conv2d(16, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, batch, arm):
        assert arm in ('features', 'image')
        probability = batch['logit'].sigmoid()
        source = batch['image'] if arm == 'image' else probability.expand(-1, 3, -1, -1)
        full = self.stem(2 * source - 1)
        low = self.image_down(F.avg_pool2d(full, 2))
        feature = self.feature_project((batch['feature'] - self.mean) / self.std)
        p_half = F.avg_pool2d(probability, 2)
        fused = self.fuse(torch.cat((feature, low, p_half, 4*p_half*(1-p_half)), 1))
        context = self.context(F.avg_pool2d(fused, 2))
        half = self.up_half(torch.cat((F.interpolate(context, scale_factor=2, mode='nearest'), fused), 1))
        raw = self.out(self.up_full(torch.cat((F.interpolate(half, scale_factor=2, mode='nearest'), full), 1)))
        mask = region_mask(probability)
        return {'probability': (batch['logit'] + mask * raw).sigmoid(),
                'full_probability': (batch['logit'] + raw).sigmoid(), 'residual': raw, 'mask': mask}

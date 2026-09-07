"""Fine visual queries preserve sub-patch variation in coarse PLAM semantics."""
import torch
from torch import nn
from torch.nn import functional as F


class DualGrainGuide(nn.Module):
    """28x28 visual queries read 14x14 context and refine a shallow PLAM map.

    Only the output projection starts at zero. It receives gradients on the
    first update; all upstream components can learn on subsequent updates.
    There is no second zero gate, auxiliary target, or auxiliary loss.
    """

    def __init__(self, channels, pool_stride, hidden=32, heads=4):
        super().__init__()
        if hidden % heads or pool_stride < 1:
            raise ValueError('Invalid hidden/head width or pooling stride')
        self.pool = nn.AvgPool2d(pool_stride, stride=pool_stride)
        self.fine_proj = nn.Conv2d(channels, hidden, 1, bias=False)
        self.fine_norm = nn.LayerNorm(hidden)
        self.coarse_norm = nn.LayerNorm(channels)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key_value = nn.Linear(channels, 2 * hidden, bias=False)
        self.local = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(), nn.Conv2d(hidden, hidden, 1),
        )
        self.output_norm = nn.LayerNorm(hidden)
        self.output_proj = nn.Conv2d(hidden, channels, 1, bias=False)
        nn.init.zeros_(self.output_proj.weight)
        self.heads = heads
        self.head_dim = hidden // heads
        self.scale = self.head_dim ** -0.5

    def forward(self, visual, coarse_tokens, guide):
        fine = self.fine_proj(self.pool(visual))
        batch, hidden, height, width = fine.shape
        if coarse_tokens.shape[1] != 196 or (height, width) != (28, 28):
            raise ValueError('P11 requires 14x14 coarse and 28x28 fine tokens')
        tokens = self.fine_norm(fine.flatten(2).transpose(1, 2))
        query = self.query(tokens).reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        key, value = self.key_value(self.coarse_norm(coarse_tokens)).chunk(2, dim=-1)
        key = key.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        value = value.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        attention = torch.softmax((query @ key.transpose(-2, -1)) * self.scale, dim=-1)
        context = (attention @ value).transpose(1, 2).reshape(batch, height * width, hidden)
        fused = (tokens + context).transpose(1, 2).reshape(batch, hidden, height, width)
        fused = fused + self.local(fused)
        fused = self.output_norm(fused.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        delta = self.output_proj(fused)
        return guide + F.interpolate(delta, size=guide.shape[-2:], mode='nearest')

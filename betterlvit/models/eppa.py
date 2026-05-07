# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class EPPA(nn.Module):
    """Frequency-Routed Edge-Preserving Attention.

    Drop-in replacement for PixLevelModule on UpblockAttention skip connections.
    Designed for fine-grained 224x224 chest X-ray segmentation where boundary
    precision dominates the Dice score.

    Background: the previous CBAM-style EPPA produced x_sa = x * ca * sa
    with ca, sa in (0, 1), so the residual was element-wise bounded by x
    and could only attenuate skip features. Diagnostics with init=0 and
    init=0.1 of the LayerScale gate both showed Adam actively pushing gate
    toward 0 (mean negative, abs_mean shrinking exponentially), because
    "scaled-down x" carried no boundary information beyond what was already
    in x. We resolve this structurally by routing high- and low-frequency
    components through separate attention branches, then recomposing.

    Architecture:
    - Frequency decomposition: depthwise 3x3 conv (one independent kernel
      per channel) initialised as a Gaussian low-pass [1,2,1;2,4,2;1,2,1]/16.
      Output x_low; high-frequency residual x_high = x - x_low.  Per-channel
      adaptation lets the low-pass cutoff differ across channels if needed,
      but it starts as a textbook Gaussian filter.
    - Channel attention from x_low: GAP + GMP -> shared MLP -> text-CLS
      logit added pre-tanh -> ca = 1 + 0.5 * tanh(ch_logit), range (0.5, 1.5).
      x_low carries semantic / region-level information that is the right
      basis for whole-channel up/down weighting.
    - Spatial attention from |x_high|: per-pixel avg/max over channels of
      the magnitude map -> 3x3 conv -> sa = 1 + tanh(sp_logit), range
      (0, 2). |x_high| is precisely an edge-magnitude map (Gaussian residual
      ~ Laplacian-of-Gaussian response), so spatial attention is fed a
      signal that is large where boundaries are and ~0 elsewhere by
      construction. sa range (0, 2) allows edge SHARPENING (sa > 1) --
      classic unsharp masking is x_low + (1 + alpha) * x_high.
    - Recomposition: x_low * ca + x_high * sa.  This is NOT bounded by x:
      sa > 1 amplifies edges; ca and sa together reshape low and high
      components independently.

    Identity at init (no gate needed):
    - low_pass.weight initialised to a Gaussian kernel (sums to 1) so x_low
      is a low-pass-filtered x, x_high is the corresponding high-pass.
    - ch_mlp[-1].weight zero-initialised -> ch_logit = 0 -> ca = 1.
    - text_proj zero-initialised (weight + bias) -> no contribution at init.
    - sp_proj.weight zero-initialised -> sp_logit = 0 -> sa = 1.
    - Output at init: x_low * 1 + x_high * 1 = x_low + (x - x_low) = x.

    The zero-init of the FINAL layers is the standard "safe-init" trick used
    in DiT / ControlNet: the last layer's own weights still receive non-zero
    gradient (grad ~ upstream * input), so they begin learning from step 1;
    earlier layers, whose gradient depends on later weights, start moving as
    soon as the last layer leaves zero. This gives "identity at init" without
    a LayerScale gate, avoiding the Adam-driven gate collapse seen in the
    previous EPPA variants.

    Resume policy: train from scratch only.  state_dict shape differs from
    both PLAM and the older CBAM-style EPPA (depthwise low_pass added,
    gate removed), so checkpoints are not interchangeable.
    """

    def __init__(self, in_channels, text_dim=None, reduction=8,
                 min_bottleneck_channels=8):
        super().__init__()
        c_red = max(in_channels // reduction,
                    min(in_channels, min_bottleneck_channels))

        # Depthwise low-pass conv, initialised to a Gaussian kernel.
        # Each channel has its own 3x3 kernel; sum=1 at init so x_low has the
        # same magnitude scale as x.  Letting it train allows per-channel
        # cutoff adaptation; we accept that the paper must argue (or verify
        # post-hoc) that the learned filters remain predominantly low-pass.
        self.low_pass = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1,
            groups=in_channels, bias=False)
        gaussian = torch.tensor([
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0],
        ]) / 16.0
        with torch.no_grad():
            self.low_pass.weight.copy_(
                gaussian.expand(in_channels, 1, 3, 3).contiguous())

        # Channel attention on x_low: shared MLP applied to GAP and GMP.
        # Last linear zero-initialised so ch_logit = 0 at init.
        self.ch_mlp = nn.Sequential(
            nn.Linear(in_channels, c_red, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_red, in_channels, bias=False),
        )
        nn.init.zeros_(self.ch_mlp[-1].weight)

        # Optional text-conditioned channel logit, added BEFORE the tanh so
        # there is a single nonlinearity in the channel path.  Zero-init so
        # text contributes 0 at start and ca = 1 regardless of text.
        self.text_dim = text_dim
        if text_dim is not None:
            self.text_proj = nn.Linear(text_dim, in_channels)
            nn.init.zeros_(self.text_proj.weight)
            nn.init.zeros_(self.text_proj.bias)

        # Spatial attention on |x_high|: 3x3 conv on [avg, max] of the
        # magnitude map.  Zero-init so sp_logit = 0 at init -> sa = 1.
        self.sp_proj = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        nn.init.zeros_(self.sp_proj.weight)

        # Diagnostic snapshot of last validation forward's ca/sa distribution.
        # Not a buffer/parameter so it stays out of state_dict (no checkpoint
        # bloat, no strict-load issue). Populated only during eval forwards.
        self._last_stats = None

    def forward(self, x, text=None):
        # 1. Frequency decomposition.
        x_low = self.low_pass(x)
        x_high = x - x_low

        # 2. Channel attention from x_low (semantic, region-level).
        avg_pool = x_low.mean(dim=(2, 3))  # [B, C]
        max_pool = x_low.amax(dim=(2, 3))  # [B, C]
        ch_logit = self.ch_mlp(avg_pool) + self.ch_mlp(max_pool)
        if text is not None and self.text_dim is not None:
            ch_logit = ch_logit + self.text_proj(text[:, 0, :])
        ca = (1.0 + 0.5 * torch.tanh(ch_logit))[:, :, None, None]  # [B, C, 1, 1]

        # 3. Spatial attention from |x_high| (boundary magnitude, sign-free).
        edge = x_high.abs()
        sp_avg = edge.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        sp_max = edge.amax(dim=1, keepdim=True)
        sa = 1.0 + torch.tanh(
            self.sp_proj(torch.cat([sp_avg, sp_max], dim=1))
        )                                                                  # [B, 1, H, W]

        # Diagnostic: stash a 5-scalar summary of ca/sa during val forwards
        # only.  Gating on not-training avoids the .item() host-device syncs
        # during training; during val we are already inside torch.no_grad()
        # and one extra sync per layer per batch is negligible (~1ms total).
        if not self.training:
            with torch.no_grad():
                self._last_stats = {
                    'ca_mean': float(ca.mean().item()),
                    'ca_std': float(ca.std().item()),
                    'sa_mean': float(sa.mean().item()),
                    'sa_std': float(sa.std().item()),
                    'sa_gt_11_ratio': float((sa > 1.1).float().mean().item()),
                }

        # 4. Recomposition: low-freq channel-modulated, high-freq spatially
        #    sharpened.  sa > 1 in boundary regions = unsharp-masking-style
        #    edge enhancement.  Output at init = x exactly.
        return x_low * ca + x_high * sa

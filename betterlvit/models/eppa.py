# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


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
      per channel) FIXED to the Gaussian low-pass [1,2,1;2,4,2;1,2,1]/16.
      Output x_low; high-frequency residual x_high = x - x_low.  Frozen
      by design: a 181-epoch diagnostic on the learnable variant (cd2903c)
      showed the per-channel kernel sum collapsed from 1.0 to ~0 across
      every stage, i.e. the conv stopped acting as a low-pass filter at
      all and x_high lost its "high-frequency residual" meaning -- which
      invalidated the sa-edge / tau-shrinkage premises this module is
      built around.  Stored as a buffer (not a Parameter), so it is not in
      `model.parameters()` / optimizer state and cannot drift.
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
    - Recomposition: x_low * ca + x_high * sa.  Not bounded by x: sa > 1
      amplifies edges (unsharp-masking-style sharpening); sa < 1 attenuates
      high-frequency content.

    Soft-thresholding on x_high (a Donoho-1995 wavelet-shrinkage analogue
    via a learnable per-channel tau_scale) was prototyped on the
    exp/eppa-smart-soft-threshold and exp/eppa-smart-st-frozen-lowpass
    branches but ablated out: with the frozen Gaussian low_pass (so x_high
    is genuine high-frequency residual), sa > 1 already takes over the
    sharpening role and sa < 1 the attenuation role -- tau_scale empirically
    settled to ~1e-4 magnitudes across all stages and was structurally
    redundant with sa.  Kept here as historical context; the live design is
    just ca + sa.

    Identity at init (no gate needed):
    - low_pass_kernel buffer is a fixed Gaussian (sums to 1) so x_low is
      a low-pass-filtered x and x_high is the corresponding high-pass.
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

        # Depthwise low-pass kernel, FROZEN to the Gaussian
        # [1,2,1;2,4,2;1,2,1]/16.  Stored as a buffer (not a Parameter) so
        # it never enters the optimizer's param list and cannot drift.  The
        # 181-epoch diagnostic on cd2903c showed the original learnable
        # variant collapsed its per-channel kernel sum from 1.0 to ~0
        # across every decoder stage (i.e. it stopped being a low-pass
        # filter), which invalidated the x_low / x_high premise the rest
        # of EPPA is built on; freezing restores that premise.
        gaussian = torch.tensor([
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0],
        ]) / 16.0
        self.register_buffer(
            'low_pass_kernel',
            gaussian.expand(in_channels, 1, 3, 3).contiguous(),
        )

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

        # Diagnostic snapshot of last forward's ca/sa distribution.
        # Not a buffer/parameter so it stays out of state_dict (no checkpoint
        # bloat, no strict-load issue). Populated unconditionally on every
        # forward -- gating on `not self.training` was tried in a prior
        # attempt and silently failed to fire (root cause unresolved; .pyc
        # cache and partial syncthing sync are leading suspects). Cost: ~5
        # .item() calls per layer per batch ~= 1ms/batch, < 0.5% overhead.
        self._last_stats = None

    def forward(self, x, text=None):
        # 1. Frequency decomposition (low_pass is the FROZEN Gaussian buffer).
        x_low = F.conv2d(
            x, self.low_pass_kernel,
            padding=1, groups=self.low_pass_kernel.shape[0],
        )
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

        # Diagnostic: stash 5-scalar ca/sa summary for the EPPA CA/SA
        # sub-table.  Runs unconditionally; cost is ~1ms/batch (negligible
        # against ~300ms forward) so training speed is unaffected.
        with torch.no_grad():
            self._last_stats = {
                'ca_mean': float(ca.mean().item()),
                'ca_std':  float(ca.std().item()),
                'sa_mean': float(sa.mean().item()),
                'sa_std':  float(sa.std().item()),
                'sa_gt_11_ratio': float((sa > 1.1).float().mean().item()),
            }

        # 4. Recomposition: low-freq channel-modulated, high-freq spatially
        #    modulated.  At init ca = sa = 1 so output = x_low + x_high = x.
        return x_low * ca + x_high * sa

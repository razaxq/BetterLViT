# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class EPPA(nn.Module):
    """Edge-Preserving Pixel Attention.

    Drop-in replacement for PixLevelModule on UpblockAttention skip connections.
    Designed for fine-grained 224x224 chest X-ray segmentation where boundary
    precision dominates the Dice score.

    Architectural decisions:
    - Channel attention: GAP -> shared MLP -> broadcast (CBAM-style, cheap).
      Per-pixel granularity is recovered via factorisation with spatial attention:
      the final per-pixel-per-channel weight is ca[c] * sa[h, w].
    - Channel-attention bottleneck floor: c_red = max(C // reduction,
      min(C, min_bottleneck_channels)). The floor (default 8) is an absolute
      minimum channel count, NOT a compression ratio. Raising it (e.g. 32)
      prevents shallow stages from collapsing to a handful of dimensions.
    - Spatial attention: 3x3 conv on [avg, max] over channels. 3x3 is the
      classic edge-detector kernel size (Sobel / Laplacian); RF = +-1 pixel
      is large enough to detect adjacent-pixel intensity gradients but
      narrow enough not to smooth them.
    - Text guidance: CXR-BERT [CLS] token (text[:, 0, :]) projected to a
      per-channel logit and ADDED to the channel-attention logit BEFORE the
      single sigmoid. One sigmoid total in the channel path -> avoids the
      double-squashing failure of (sigmoid(a) * sigmoid(b)) ~ 0.25 collapse.
      [CLS] is used (not masked-mean) because CXR-BERT-specialized's [CLS]
      is image-text contrastive aligned during pretraining and is by
      construction free of [PAD] pollution.
    - Residual: per-channel LayerScale-style learnable parameter
      (1, C, 1, 1), init = 0.1 (NOT 0; see __init__ comment for the
      diagnostic that motivated this). Each channel still learns its
      own positive or negative contribution; no activation function on
      the gate so gradient flow is direct.

    Resume policy: train from scratch only. Loading a PLAM checkpoint into
    an EPPA model causes decoder distribution shift (decoder was trained
    expecting PLAM-modulated skip features; EPPA at init produces a
    small random perturbation -- gate=0.1 amplifies untrained ch_mlp /
    sp_proj output by 10% and adds it to the skip).
    """

    def __init__(self, in_channels, text_dim=None, reduction=8,
                 min_bottleneck_channels=8):
        super().__init__()
        c_red = max(in_channels // reduction,
                    min(in_channels, min_bottleneck_channels))

        # Channel attention: GAP -> shared MLP. avg and max pool both pass
        # through the SAME MLP; results are summed (CBAM convention).
        self.ch_mlp = nn.Sequential(
            nn.Linear(in_channels, c_red, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_red, in_channels, bias=False),
        )

        # Optional text-conditioned channel logit.
        self.text_dim = text_dim
        if text_dim is not None:
            self.text_proj = nn.Linear(text_dim, in_channels)

        # Spatial attention: 3x3 conv on [avg, max] over channels.
        self.sp_proj = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)

        # Per-channel LayerScale-style residual gate, init = 0.1 (NOT 0).
        # init=0 was diagnosed dead in epoch 1-9 (see train_model.py gate
        # sub-table): abs_mean < 0.01 and trending DOWN, mean systematically
        # negative -- Adam was actively closing the gate. 0.1 forces a
        # non-trivial x_sa residual contribution from step 1 so inner
        # ch_mlp / sp_proj receive learnable-magnitude gradients before
        # the cold-start coupling traps them at zero.
        self.gate = nn.Parameter(torch.full((1, in_channels, 1, 1), 0.1))

    def forward(self, x, text=None):
        # Channel attention via global avg / max pool through shared MLP.
        avg_pool = x.mean(dim=(2, 3))                                     # [B, C]
        max_pool = x.amax(dim=(2, 3))                                     # [B, C]
        ch_logit = self.ch_mlp(avg_pool) + self.ch_mlp(max_pool)          # [B, C]

        # Add text logit BEFORE sigmoid -- only one sigmoid in this path,
        # no double-squashing.
        if text is not None and self.text_dim is not None:
            text_cls = text[:, 0, :]                                      # [B, text_dim]
            ch_logit = ch_logit + self.text_proj(text_cls)

        ca = torch.sigmoid(ch_logit)[:, :, None, None]                    # [B, C, 1, 1]
        x_ca = x * ca

        # Spatial attention with 3x3 conv (edge-aware, minimal smoothing).
        sp_avg = x_ca.mean(dim=1, keepdim=True)                           # [B, 1, H, W]
        sp_max = x_ca.amax(dim=1, keepdim=True)                           # [B, 1, H, W]
        sa = torch.sigmoid(
            self.sp_proj(torch.cat([sp_avg, sp_max], dim=1))
        )                                                                  # [B, 1, H, W]
        x_sa = x_ca * sa

        # Per-channel LayerScale residual.
        return x + self.gate * x_sa

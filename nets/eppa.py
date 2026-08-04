# -*- coding: utf-8 -*-
"""Decoder-guided frequency-routed attention for LViT skip connections."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderGuidedEPPA(nn.Module):
    """Decoder-Guided Multi-Scale Edge-Preserving Pixel Attention.

    Vanilla PLAM refines an encoder skip from that skip alone. FreqEPPA adds a
    useful frequency prior, but its spatial gain still cannot distinguish a
    task-relevant boundary from an irrelevant high-frequency structure. This
    module injects the corresponding decoder feature as a coarse semantic
    gating signal while retaining the frequency-routed EPPA decomposition.

    The spatial branch fuses four signals at the skip resolution:
    low-frequency skip semantics, decoder semantics, local edge evidence, and
    dilated edge context. A zero-initialised output projection makes the
    spatial gain exactly one at initialisation, so the whole module remains an
    identity mapping at step zero.
    """

    def __init__(
        self,
        in_channels,
        text_dim=None,
        reduction=8,
        min_bottleneck_channels=8,
        guide_reduction=8,
        min_guide_channels=16,
        use_decoder_guide=True,
        use_dilated_edge=True,
    ):
        super().__init__()
        channel_bottleneck = max(
            in_channels // reduction,
            min(in_channels, min_bottleneck_channels),
        )
        guide_channels = max(
            in_channels // guide_reduction,
            min(in_channels, min_guide_channels),
        )

        self.low_pass = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        gaussian = torch.tensor(
            [
                [1.0, 2.0, 1.0],
                [2.0, 4.0, 2.0],
                [1.0, 2.0, 1.0],
            ]
        ) / 16.0
        with torch.no_grad():
            self.low_pass.weight.copy_(
                gaussian.expand(
                    in_channels,
                    1,
                    3,
                    3,
                ).contiguous()
            )

        self.channel_mlp = nn.Sequential(
            nn.Linear(
                in_channels,
                channel_bottleneck,
                bias=False,
            ),
            nn.SiLU(inplace=True),
            nn.Linear(
                channel_bottleneck,
                in_channels,
                bias=False,
            ),
        )
        nn.init.zeros_(self.channel_mlp[-1].weight)

        self.text_dim = text_dim
        self.use_decoder_guide = use_decoder_guide
        self.use_dilated_edge = use_dilated_edge
        if text_dim is not None:
            self.text_channel_proj = nn.Linear(
                text_dim,
                in_channels,
            )
            nn.init.zeros_(self.text_channel_proj.weight)
            nn.init.zeros_(self.text_channel_proj.bias)

        self.skip_semantic_proj = nn.Conv2d(
            in_channels,
            guide_channels,
            kernel_size=1,
            bias=False,
        )
        self.decoder_semantic_proj = nn.Conv2d(
            in_channels,
            guide_channels,
            kernel_size=1,
            bias=False,
        )
        self.edge_local = nn.Conv2d(
            2,
            guide_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.edge_context = nn.Conv2d(
            2,
            guide_channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            bias=False,
        )
        self.guide_norm = nn.GroupNorm(
            num_groups=1,
            num_channels=guide_channels,
        )
        self.guide_activation = nn.SiLU(inplace=True)
        self.spatial_out = nn.Conv2d(
            guide_channels,
            1,
            kernel_size=1,
            bias=True,
        )
        nn.init.zeros_(self.spatial_out.weight)
        nn.init.zeros_(self.spatial_out.bias)

        self._last_stats = None

    def _decoder_at_skip_resolution(self, decoder, skip):
        if decoder is None:
            return torch.zeros_like(skip)
        if decoder.shape[1] != skip.shape[1]:
            raise ValueError(
                "Decoder and skip channel counts must match: "
                f"{decoder.shape[1]} != {skip.shape[1]}"
            )
        if decoder.shape[-2:] != skip.shape[-2:]:
            decoder = F.interpolate(
                decoder,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return decoder

    def forward(self, skip, decoder=None, text=None):
        decoder = self._decoder_at_skip_resolution(
            decoder,
            skip,
        )

        skip_low = self.low_pass(skip)
        skip_high = skip - skip_low

        average_pool = skip_low.mean(dim=(2, 3))
        maximum_pool = skip_low.amax(dim=(2, 3))
        channel_logit = (
            self.channel_mlp(average_pool)
            + self.channel_mlp(maximum_pool)
        )
        if text is not None and self.text_dim is not None:
            channel_logit = (
                channel_logit
                + self.text_channel_proj(text[:, 0, :])
            )
        channel_gain = (
            1.0 + 0.5 * torch.tanh(channel_logit)
        )[:, :, None, None]

        edge_magnitude = skip_high.abs()
        edge_statistics = torch.cat(
            [
                edge_magnitude.mean(dim=1, keepdim=True),
                edge_magnitude.amax(dim=1, keepdim=True),
            ],
            dim=1,
        )
        guide_features = (
            self.skip_semantic_proj(skip_low)
            + self.edge_local(edge_statistics)
        )
        if self.use_decoder_guide:
            guide_features = (
                guide_features
                + self.decoder_semantic_proj(decoder)
            )
        if self.use_dilated_edge:
            guide_features = (
                guide_features
                + self.edge_context(edge_statistics)
            )
        guide_features = self.guide_activation(
            self.guide_norm(guide_features)
        )
        spatial_logit = self.spatial_out(guide_features)
        spatial_gain = 1.0 + torch.tanh(spatial_logit)

        output = (
            skip_low * channel_gain
            + skip_high * spatial_gain
        )

        if not self.training:
            with torch.no_grad():
                self._last_stats = {
                    "channel_mean": float(
                        channel_gain.mean().item()
                    ),
                    "channel_std": float(
                        channel_gain.std().item()
                    ),
                    "spatial_mean": float(
                        spatial_gain.mean().item()
                    ),
                    "spatial_std": float(
                        spatial_gain.std().item()
                    ),
                    "spatial_amplify_ratio": float(
                        (spatial_gain > 1.1).float().mean().item()
                    ),
                    "spatial_suppress_ratio": float(
                        (spatial_gain < 0.9).float().mean().item()
                    ),
                    "guide_abs_mean": float(
                        guide_features.abs().mean().item()
                    ),
                }
        return output


# Keep the public name used by the existing LViT import path.
EPPA = DecoderGuidedEPPA

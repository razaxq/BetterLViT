# -*- coding: utf-8 -*-
"""Balanced residual decoder-guided attention for LViT skip connections."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedResidualDGEPPA(nn.Module):
    """Balanced-Residual Decoder-Guided EPPA.

    The first DG-EPPA experiment improved over tag 839752, but its spatial
    branch converged to near-global high-frequency suppression at three of the
    four decoder stages. This version retains the useful frequency routing and
    decoder semantics while constraining the attention to a residual highway.

    A zero-mean local mask can redistribute high-frequency evidence without
    changing its spatial mean. A separate, tightly bounded global mask can
    still suppress noisy high frequencies when justified. Both paths use
    learnable per-channel strengths and preserve the unmodified skip as the
    main information highway. The spatial and channel heads are
    zero-initialised, making the complete module an identity at step zero.
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
        balance_spatial=True,
        use_text_spatial_film=True,
        normalize_channel_descriptors=True,
        local_strength_max=0.5,
        global_strength_max=0.15,
        local_strength_init=0.1,
        global_strength_init=0.05,
    ):
        super().__init__()
        if not 0.0 < local_strength_init < local_strength_max:
            raise ValueError(
                "local_strength_init must be between zero and its maximum"
            )
        if not 0.0 < global_strength_init < global_strength_max:
            raise ValueError(
                "global_strength_init must be between zero and its maximum"
            )
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
        self.balance_spatial = balance_spatial
        self.use_text_spatial_film = use_text_spatial_film
        self.normalize_channel_descriptors = normalize_channel_descriptors
        self.local_strength_max = float(local_strength_max)
        self.global_strength_max = float(global_strength_max)
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
        self.guide_branch_logits = nn.Parameter(
            torch.zeros(4, guide_channels)
        )
        self.guide_norm = nn.GroupNorm(
            num_groups=1,
            num_channels=guide_channels,
        )
        if text_dim is not None:
            self.text_spatial_film = nn.Linear(
                text_dim,
                guide_channels * 2,
            )
            nn.init.zeros_(self.text_spatial_film.weight)
            nn.init.zeros_(self.text_spatial_film.bias)
        self.guide_activation = nn.SiLU(inplace=True)
        self.spatial_out = nn.Conv2d(
            guide_channels,
            1,
            kernel_size=1,
            bias=True,
        )
        nn.init.zeros_(self.spatial_out.weight)
        nn.init.zeros_(self.spatial_out.bias)

        local_ratio = local_strength_init / local_strength_max
        global_ratio = global_strength_init / global_strength_max
        self.local_strength_logit = nn.Parameter(
            torch.full(
                (1, in_channels, 1, 1),
                math.log(local_ratio / (1.0 - local_ratio)),
            )
        )
        self.global_strength_logit = nn.Parameter(
            torch.full(
                (1, in_channels, 1, 1),
                math.log(global_ratio / (1.0 - global_ratio)),
            )
        )

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
        if self.normalize_channel_descriptors:
            descriptor_scale = math.sqrt(skip.shape[1])
            average_pool = F.normalize(
                average_pool,
                p=2,
                dim=1,
                eps=1e-6,
            ) * descriptor_scale
            maximum_pool = F.normalize(
                maximum_pool,
                p=2,
                dim=1,
                eps=1e-6,
            ) * descriptor_scale
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
        skip_guide = self.skip_semantic_proj(skip_low)
        decoder_guide = self.decoder_semantic_proj(decoder)
        if not self.use_decoder_guide:
            decoder_guide = torch.zeros_like(decoder_guide)
        edge_local = self.edge_local(edge_statistics)
        edge_context = self.edge_context(edge_statistics)
        if not self.use_dilated_edge:
            edge_context = torch.zeros_like(edge_context)
        guide_branch_weights = torch.softmax(
            self.guide_branch_logits,
            dim=0,
        )
        guide_features = torch.zeros_like(skip_guide)
        for branch_index, branch_features in enumerate(
            (
                skip_guide,
                decoder_guide,
                edge_local,
                edge_context,
            )
        ):
            guide_features = (
                guide_features
                + branch_features
                * guide_branch_weights[
                    branch_index
                ][None, :, None, None]
            )
        guide_features = self.guide_norm(guide_features)

        text_film_magnitude = None
        if (
            text is not None
            and self.text_dim is not None
            and self.use_text_spatial_film
        ):
            film = self.text_spatial_film(text[:, 0, :])
            film_scale, film_bias = film.chunk(2, dim=1)
            film_scale = 0.25 * torch.tanh(film_scale)
            film_bias = 0.25 * torch.tanh(film_bias)
            guide_features = (
                guide_features
                * (1.0 + film_scale[:, :, None, None])
                + film_bias[:, :, None, None]
            )
            if not self.training:
                text_film_magnitude = (
                    torch.cat([film_scale, film_bias], dim=1)
                    .abs()
                    .mean()
                )

        guide_features = self.guide_activation(guide_features)
        spatial_logit = self.spatial_out(guide_features)
        signed_spatial = torch.tanh(spatial_logit)
        spatial_global = signed_spatial.mean(
            dim=(2, 3),
            keepdim=True,
        )
        if self.balance_spatial:
            # Multiplication by 0.5 keeps the exactly zero-mean local residual
            # in [-1, 1] because signed_spatial itself lies in [-1, 1].
            spatial_local = 0.5 * (
                signed_spatial - spatial_global
            )
        else:
            spatial_local = signed_spatial

        local_strength = (
            self.local_strength_max
            * torch.sigmoid(self.local_strength_logit)
        )
        global_strength = (
            self.global_strength_max
            * torch.sigmoid(self.global_strength_logit)
        )
        spatial_residual = (
            spatial_local * local_strength
            + spatial_global * global_strength
        )
        spatial_gain = 1.0 + spatial_residual

        # Written as a residual update to make the unmodified skip an explicit
        # information highway. At initialisation both residuals are zero.
        output = (
            skip
            + skip_low * (channel_gain - 1.0)
            + skip_high * spatial_residual
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
                    "spatial_min": float(
                        spatial_gain.amin().item()
                    ),
                    "spatial_max": float(
                        spatial_gain.amax().item()
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
                    "spatial_local_mean": float(
                        spatial_local.mean().item()
                    ),
                    "spatial_global_mean": float(
                        spatial_global.mean().item()
                    ),
                    "local_strength_mean": float(
                        local_strength.mean().item()
                    ),
                    "global_strength_mean": float(
                        global_strength.mean().item()
                    ),
                    "spatial_saturation_ratio": float(
                        (signed_spatial.abs() > 0.95)
                        .float()
                        .mean()
                        .item()
                    ),
                    "text_film_abs_mean": (
                        float(text_film_magnitude.item())
                        if text_film_magnitude is not None
                        else 0.0
                    ),
                    "guide_branch_entropy": float(
                        (
                            -guide_branch_weights
                            * torch.log(
                                guide_branch_weights.clamp_min(1e-8)
                            )
                        )
                        .sum(dim=0)
                        .mean()
                        .div(math.log(4.0))
                        .item()
                    ),
                    "guide_skip_weight": float(
                        guide_branch_weights[0].mean().item()
                    ),
                    "guide_decoder_weight": float(
                        guide_branch_weights[1].mean().item()
                    ),
                    "guide_local_edge_weight": float(
                        guide_branch_weights[2].mean().item()
                    ),
                    "guide_context_edge_weight": float(
                        guide_branch_weights[3].mean().item()
                    ),
                }
        return output


# Keep the public name used by the existing LViT import path.
EPPA = BalancedResidualDGEPPA

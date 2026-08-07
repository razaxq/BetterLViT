# -*- coding: utf-8 -*-
"""PLAM-guided, normalized frequency routing for LViT skip connections."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalizedLowPass(nn.Module):
    """Per-channel non-negative 3x3 kernels constrained to sum to one."""

    def __init__(self, channels):
        super().__init__()
        gaussian = torch.tensor(
            [
                [1.0, 2.0, 1.0],
                [2.0, 4.0, 2.0],
                [1.0, 2.0, 1.0],
            ],
            dtype=torch.float32,
        ) / 16.0
        initial_kernel = gaussian.view(1, 1, 3, 3)
        self.kernel_logits = nn.Parameter(
            initial_kernel.log().expand(channels, 1, 3, 3).clone()
        )
        self.register_buffer("initial_kernel", initial_kernel)
        self.channels = channels

    def kernel(self):
        return torch.softmax(
            self.kernel_logits.flatten(2),
            dim=-1,
        ).view(self.channels, 1, 3, 3)

    def forward(self, inputs):
        padded = F.pad(inputs, (1, 1, 1, 1), mode="reflect")
        return F.conv2d(
            padded,
            self.kernel(),
            groups=self.channels,
        )


class PLAMGuidedNormalizedEPPA(nn.Module):
    """Pixel-semantic and frequency-residual attention with identity init.

    The BR-DG-EPPA experiment showed that a four-way static softmax stayed
    almost uniform while the shallow spatial paths collapsed. This version
    removes that competition. A PLAM-inspired semantic path receives the skip,
    decoder and text features, while a separate EPPA path models local and
    contextual high-frequency evidence. Each path owns a direct zero-initialised
    output head, so useful gradients are not mediated by a shared branch gate.

    The frequency decomposition is constrained: every depthwise low-pass
    kernel remains non-negative and sums to one throughout training. The full
    block is exactly the identity at initialisation.
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
        use_text_pixel_film=True,
        normalize_channel_descriptors=True,
        channel_strength_max=0.5,
        pixel_strength_max=0.35,
        edge_strength_max=0.5,
        pixel_strength_init=0.1,
        edge_strength_init=0.15,
    ):
        super().__init__()
        if not 0.0 < pixel_strength_init < pixel_strength_max:
            raise ValueError(
                "pixel_strength_init must be between zero and its maximum"
            )
        if not 0.0 < edge_strength_init < edge_strength_max:
            raise ValueError(
                "edge_strength_init must be between zero and its maximum"
            )

        channel_bottleneck = max(
            in_channels // reduction,
            min(in_channels, min_bottleneck_channels),
        )
        guide_channels = max(
            in_channels // guide_reduction,
            min(in_channels, min_guide_channels),
        )
        pixel_hidden = max(8, guide_channels // 2)

        self.low_pass = NormalizedLowPass(in_channels)
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
        self.use_text_pixel_film = use_text_pixel_film
        self.normalize_channel_descriptors = (
            normalize_channel_descriptors
        )
        self.channel_strength_max = float(channel_strength_max)
        self.pixel_strength_max = float(pixel_strength_max)
        self.edge_strength_max = float(edge_strength_max)

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
        self.skip_semantic_norm = nn.GroupNorm(
            num_groups=1,
            num_channels=guide_channels,
        )
        self.decoder_semantic_norm = nn.GroupNorm(
            num_groups=1,
            num_channels=guide_channels,
        )
        if text_dim is not None:
            self.text_pixel_film = nn.Linear(
                text_dim,
                guide_channels * 2,
            )
            nn.init.zeros_(self.text_pixel_film.weight)
            nn.init.zeros_(self.text_pixel_film.bias)

        # The first three descriptors reproduce PLAM's average, maximum and
        # their sum. Decoder-skip cosine agreement adds top-down semantics.
        self.pixel_mlp = nn.Sequential(
            nn.Conv2d(4, pixel_hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(pixel_hidden, 1, kernel_size=1),
        )
        nn.init.zeros_(self.pixel_mlp[-1].weight)
        nn.init.zeros_(self.pixel_mlp[-1].bias)

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
        self.edge_norm = nn.GroupNorm(
            num_groups=1,
            num_channels=guide_channels,
        )
        self.edge_activation = nn.SiLU(inplace=True)
        self.edge_out = nn.Conv2d(
            guide_channels,
            1,
            kernel_size=1,
        )
        nn.init.zeros_(self.edge_out.weight)
        nn.init.zeros_(self.edge_out.bias)

        pixel_ratio = pixel_strength_init / pixel_strength_max
        edge_ratio = edge_strength_init / edge_strength_max
        self.pixel_strength_logit = nn.Parameter(
            torch.tensor(
                math.log(pixel_ratio / (1.0 - pixel_ratio))
            ).view(1, 1, 1, 1)
        )
        self.edge_strength_logit = nn.Parameter(
            torch.tensor(
                math.log(edge_ratio / (1.0 - edge_ratio))
            ).view(1, 1, 1, 1)
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

    def _channel_gain(self, skip_low, text):
        average_pool = skip_low.mean(dim=(2, 3))
        maximum_pool = skip_low.amax(dim=(2, 3))
        if self.normalize_channel_descriptors:
            descriptor_scale = math.sqrt(skip_low.shape[1])
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
        return (
            1.0
            + self.channel_strength_max * torch.tanh(channel_logit)
        )[:, :, None, None]

    def _semantic_pixel_residual(self, skip_low, decoder, text):
        skip_features = self.skip_semantic_norm(
            self.skip_semantic_proj(skip_low)
        )
        decoder_features = self.decoder_semantic_norm(
            self.decoder_semantic_proj(decoder)
        )
        if not self.use_decoder_guide:
            decoder_features = torch.zeros_like(decoder_features)

        semantic_features = skip_features + decoder_features
        text_film_magnitude = None
        if (
            text is not None
            and self.text_dim is not None
            and self.use_text_pixel_film
        ):
            film = self.text_pixel_film(text[:, 0, :])
            film_scale, film_bias = film.chunk(2, dim=1)
            film_scale = 0.25 * torch.tanh(film_scale)
            film_bias = 0.25 * torch.tanh(film_bias)
            semantic_features = (
                semantic_features
                * (1.0 + film_scale[:, :, None, None])
                + film_bias[:, :, None, None]
            )
            if not self.training:
                text_film_magnitude = (
                    torch.cat([film_scale, film_bias], dim=1)
                    .abs()
                    .mean()
                )

        semantic_features = F.silu(semantic_features)
        semantic_average = semantic_features.mean(
            dim=1,
            keepdim=True,
        )
        semantic_maximum = semantic_features.amax(
            dim=1,
            keepdim=True,
        )
        skip_normalized = F.normalize(
            skip_features,
            p=2,
            dim=1,
            eps=1e-6,
        )
        decoder_normalized = F.normalize(
            decoder_features,
            p=2,
            dim=1,
            eps=1e-6,
        )
        agreement = (
            skip_normalized * decoder_normalized
        ).sum(dim=1, keepdim=True)
        descriptors = torch.cat(
            [
                semantic_average,
                semantic_maximum,
                semantic_average + semantic_maximum,
                agreement,
            ],
            dim=1,
        )
        pixel_logit = self.pixel_mlp(descriptors)
        return (
            torch.tanh(pixel_logit),
            pixel_logit,
            skip_features,
            decoder_features,
            semantic_features,
            text_film_magnitude,
        )

    def _edge_residual(self, skip_high, pixel_logit):
        edge_magnitude = skip_high.abs()
        edge_statistics = torch.cat(
            [
                edge_magnitude.mean(dim=1, keepdim=True),
                edge_magnitude.amax(dim=1, keepdim=True),
            ],
            dim=1,
        )
        local_features = self.edge_local(edge_statistics)
        context_features = self.edge_context(edge_statistics)
        if not self.use_dilated_edge:
            context_features = torch.zeros_like(context_features)
        edge_features = self.edge_activation(
            self.edge_norm(local_features + context_features)
        )
        edge_logit = self.edge_out(edge_features)
        semantic_support = torch.sigmoid(pixel_logit)
        edge_residual = (
            torch.tanh(edge_logit) * semantic_support
        )
        return (
            edge_residual,
            semantic_support,
            local_features,
            context_features,
            edge_features,
        )

    def forward(self, skip, decoder=None, text=None):
        decoder = self._decoder_at_skip_resolution(decoder, skip)
        skip_low = self.low_pass(skip)
        skip_high = skip - skip_low
        channel_gain = self._channel_gain(skip_low, text)

        (
            pixel_residual,
            pixel_logit,
            skip_features,
            decoder_features,
            semantic_features,
            text_film_magnitude,
        ) = self._semantic_pixel_residual(skip_low, decoder, text)
        (
            edge_residual,
            semantic_support,
            local_features,
            context_features,
            edge_features,
        ) = self._edge_residual(skip_high, pixel_logit)

        pixel_strength = (
            self.pixel_strength_max
            * torch.sigmoid(self.pixel_strength_logit)
        )
        edge_strength = (
            self.edge_strength_max
            * torch.sigmoid(self.edge_strength_logit)
        )
        spatial_residual = (
            pixel_strength * pixel_residual
            + edge_strength * edge_residual
        )
        spatial_gain = 1.0 + spatial_residual

        output = (
            skip
            + skip_low * (channel_gain - 1.0)
            + skip * pixel_strength * pixel_residual
            + skip_high * edge_strength * edge_residual
        )

        if not self.training:
            with torch.no_grad():
                branch_energy = torch.stack(
                    [
                        skip_features.abs().mean(),
                        decoder_features.abs().mean(),
                        local_features.abs().mean(),
                        context_features.abs().mean(),
                    ]
                ).clamp_min(1e-8)
                branch_weights = branch_energy / branch_energy.sum()
                low_pass_kernel = self.low_pass.kernel()
                kernel_entropy = (
                    -low_pass_kernel
                    * low_pass_kernel.clamp_min(1e-8).log()
                ).sum(dim=(2, 3)).mean() / math.log(9.0)
                kernel_delta = (
                    low_pass_kernel
                    - self.low_pass.initial_kernel
                ).abs().mean()
                self._last_stats = {
                    "channel_mean": float(channel_gain.mean().item()),
                    "channel_std": float(channel_gain.std().item()),
                    "spatial_mean": float(spatial_gain.mean().item()),
                    "spatial_std": float(spatial_gain.std().item()),
                    "spatial_min": float(spatial_gain.amin().item()),
                    "spatial_max": float(spatial_gain.amax().item()),
                    "spatial_amplify_ratio": float(
                        (spatial_gain > 1.1).float().mean().item()
                    ),
                    "spatial_suppress_ratio": float(
                        (spatial_gain < 0.9).float().mean().item()
                    ),
                    "guide_abs_mean": float(
                        0.5
                        * (
                            semantic_features.abs().mean()
                            + edge_features.abs().mean()
                        ).item()
                    ),
                    # Legacy names are retained for existing history readers.
                    "spatial_local_mean": float(
                        pixel_residual.mean().item()
                    ),
                    "spatial_global_mean": float(
                        edge_residual.mean().item()
                    ),
                    "local_strength_mean": float(
                        pixel_strength.mean().item()
                    ),
                    "global_strength_mean": float(
                        edge_strength.mean().item()
                    ),
                    "spatial_saturation_ratio": float(
                        0.5
                        * (
                            (pixel_residual.abs() > 0.95).float().mean()
                            + (edge_residual.abs() > 0.95).float().mean()
                        ).item()
                    ),
                    "text_film_abs_mean": (
                        float(text_film_magnitude.item())
                        if text_film_magnitude is not None
                        else 0.0
                    ),
                    "guide_branch_entropy": float(
                        (
                            -branch_weights
                            * branch_weights.clamp_min(1e-8).log()
                        ).sum().div(math.log(4.0)).item()
                    ),
                    "guide_skip_weight": float(branch_weights[0].item()),
                    "guide_decoder_weight": float(
                        branch_weights[1].item()
                    ),
                    "guide_local_edge_weight": float(
                        branch_weights[2].item()
                    ),
                    "guide_context_edge_weight": float(
                        branch_weights[3].item()
                    ),
                    "pixel_residual_std": float(
                        pixel_residual.std().item()
                    ),
                    "edge_residual_std": float(
                        edge_residual.std().item()
                    ),
                    "semantic_support_mean": float(
                        semantic_support.mean().item()
                    ),
                    "low_pass_kernel_sum": float(
                        low_pass_kernel.sum(dim=(2, 3)).mean().item()
                    ),
                    "low_pass_kernel_entropy": float(
                        kernel_entropy.item()
                    ),
                    "low_pass_kernel_center": float(
                        low_pass_kernel[:, :, 1, 1].mean().item()
                    ),
                    "low_pass_kernel_delta_abs": float(
                        kernel_delta.item()
                    ),
                }
        return output


# Keep the public name used by the existing LViT import path.
EPPA = PLAMGuidedNormalizedEPPA

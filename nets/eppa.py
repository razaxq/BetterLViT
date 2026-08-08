# -*- coding: utf-8 -*-
"""Frequency-aligned PLAM-guided skip refinement for BetterLViT."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedHaarFrequencySplit(nn.Module):
    """Exactly decompose a feature map into Haar low/high reconstructions.

    The filters are fixed buffers rather than trainable logits. Consequently,
    weight decay cannot turn the decomposition into the uniform 3x3 averaging
    kernel observed in the previous normalized-EPPA experiment.
    """

    def __init__(self, channels):
        super().__init__()
        filters = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[-1.0, -1.0], [1.0, 1.0]],
                [[-1.0, 1.0], [-1.0, 1.0]],
                [[1.0, -1.0], [-1.0, 1.0]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1) / 2.0
        self.register_buffer(
            "filters",
            filters.repeat(channels, 1, 1, 1),
        )
        self.channels = int(channels)

    def _analysis(self, inputs):
        height, width = inputs.shape[-2:]
        pad_height = height % 2
        pad_width = width % 2
        if pad_height or pad_width:
            inputs = F.pad(
                inputs,
                (0, pad_width, 0, pad_height),
                mode="replicate",
            )
        coefficients = F.conv2d(
            inputs,
            self.filters,
            stride=2,
            groups=self.channels,
        )
        batch, _, coeff_height, coeff_width = coefficients.shape
        coefficients = coefficients.view(
            batch,
            self.channels,
            4,
            coeff_height,
            coeff_width,
        )
        return coefficients, (height, width)

    def _synthesis(self, coefficients, output_size):
        batch, channels, bands, height, width = coefficients.shape
        if channels != self.channels or bands != 4:
            raise ValueError(
                "Expected Haar coefficients [B, {}, 4, H, W], got {}".format(
                    self.channels,
                    tuple(coefficients.shape),
                )
            )
        reconstructed = F.conv_transpose2d(
            coefficients.reshape(batch, channels * bands, height, width),
            self.filters,
            stride=2,
            groups=self.channels,
        )
        output_height, output_width = output_size
        return reconstructed[:, :, :output_height, :output_width]

    def forward(self, inputs):
        if inputs.shape[1] != self.channels:
            raise ValueError(
                "Expected {} channels, got {}".format(
                    self.channels,
                    inputs.shape[1],
                )
            )
        coefficients, output_size = self._analysis(inputs)
        low_coefficients = torch.zeros_like(coefficients)
        low_coefficients[:, :, 0] = coefficients[:, :, 0]
        high_coefficients = coefficients - low_coefficients
        low = self._synthesis(low_coefficients, output_size)
        high = self._synthesis(high_coefficients, output_size)
        return low, high


def _strength_logit(initial, maximum, floor=0.0):
    if not floor <= initial < maximum:
        raise ValueError(
            "Strength initial value must satisfy floor <= initial < maximum"
        )
    ratio = (initial - floor) / (maximum - floor)
    ratio = min(max(ratio, 1e-4), 1.0 - 1e-4)
    return math.log(ratio / (1.0 - ratio))


class FAMHaarEPPA(nn.Module):
    """FAM-EPPA V4-A: separated PLAM semantics and stable Haar detail.

    Raw CNN, PLAM and decoder features remain separate until this block. The
    raw skip is decomposed by a fixed, exactly reconstructing Haar transform.
    Its low-frequency component feeds the semantic/channel path, while its
    high-frequency component owns a direct additive residual path with a small
    non-zero floor. PLAM contributes only its low-frequency reconstruction as a
    semantic residual, avoiding the irreversible pre-EPPA addition used by V3.

    V4-A intentionally retains the existing CLS-token FiLM and same-scale
    decoder guide. Token-level LFFI, adaptive filters and cross-scale routing
    belong to later, separately measurable ablations.
    """

    architecture_version = "fam_eppa_v4a"

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
        edge_strength_max=0.30,
        pixel_strength_init=0.10,
        edge_strength_init=0.10,
        use_plam_guide=True,
        plam_strength_max=1.25,
        plam_strength_init=1.0,
        plam_strength_floor=0.25,
        detail_strength_floor=0.02,
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
        pixel_hidden = max(8, guide_channels // 2)

        self.in_channels = int(in_channels)
        self.text_dim = text_dim
        self.use_decoder_guide = bool(use_decoder_guide)
        self.use_dilated_detail = bool(use_dilated_edge)
        self.use_text_pixel_film = bool(use_text_pixel_film)
        self.use_plam_guide = bool(use_plam_guide)
        self.normalize_channel_descriptors = bool(
            normalize_channel_descriptors
        )
        self.channel_strength_max = float(channel_strength_max)
        self.region_strength_max = float(pixel_strength_max)
        self.detail_strength_max = float(edge_strength_max)
        self.detail_strength_floor = float(detail_strength_floor)
        self.plam_strength_max = float(plam_strength_max)
        self.plam_strength_floor = float(plam_strength_floor)

        self.frequency_split = FixedHaarFrequencySplit(in_channels)

        self.channel_mlp = nn.Sequential(
            nn.Linear(in_channels, channel_bottleneck, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channel_bottleneck, in_channels, bias=False),
        )
        nn.init.zeros_(self.channel_mlp[-1].weight)

        if text_dim is not None:
            self.text_channel_proj = nn.Linear(text_dim, in_channels)
            nn.init.zeros_(self.text_channel_proj.weight)
            nn.init.zeros_(self.text_channel_proj.bias)

        self.skip_semantic_proj = nn.Conv2d(
            in_channels,
            guide_channels,
            kernel_size=1,
            bias=False,
        )
        self.plam_semantic_proj = nn.Conv2d(
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
        self.skip_semantic_norm = nn.GroupNorm(1, guide_channels)
        self.plam_semantic_norm = nn.GroupNorm(1, guide_channels)
        self.decoder_semantic_norm = nn.GroupNorm(1, guide_channels)

        if text_dim is not None:
            self.text_pixel_film = nn.Linear(
                text_dim,
                guide_channels * 2,
            )
            nn.init.zeros_(self.text_pixel_film.weight)
            nn.init.zeros_(self.text_pixel_film.bias)

        self.region_spatial = nn.Sequential(
            nn.Conv2d(4, pixel_hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(pixel_hidden, 1, kernel_size=1),
        )
        nn.init.zeros_(self.region_spatial[-1].weight)
        nn.init.zeros_(self.region_spatial[-1].bias)
        self.region_out = nn.Sequential(
            nn.Conv2d(
                guide_channels,
                guide_channels,
                kernel_size=3,
                padding=1,
                groups=guide_channels,
                bias=False,
            ),
            nn.GroupNorm(1, guide_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(guide_channels, in_channels, kernel_size=1),
        )
        nn.init.zeros_(self.region_out[-1].weight)
        nn.init.zeros_(self.region_out[-1].bias)

        self.detail_local = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.detail_context = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=in_channels,
            bias=False,
        )
        detail_groups = min(8, in_channels)
        while in_channels % detail_groups:
            detail_groups -= 1
        self.detail_norm = nn.GroupNorm(detail_groups, in_channels)
        self.detail_out = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
        )
        nn.init.zeros_(self.detail_out.weight)
        nn.init.zeros_(self.detail_out.bias)

        self.region_strength_logit = nn.Parameter(
            torch.tensor(
                _strength_logit(
                    pixel_strength_init,
                    self.region_strength_max,
                )
            ).view(1, 1, 1, 1)
        )
        self.detail_strength_logit = nn.Parameter(
            torch.tensor(
                _strength_logit(
                    edge_strength_init,
                    self.detail_strength_max,
                    self.detail_strength_floor,
                )
            ).view(1, 1, 1, 1)
        )
        self.plam_strength_logit = nn.Parameter(
            torch.tensor(
                _strength_logit(
                    plam_strength_init,
                    self.plam_strength_max,
                    self.plam_strength_floor,
                )
            ).view(1, 1, 1, 1)
        )
        self._last_stats = None

    @staticmethod
    def _at_skip_resolution(features, skip, name):
        if features is None:
            return torch.zeros_like(skip)
        if features.shape[1] != skip.shape[1]:
            raise ValueError(
                "{} and skip channel counts must match: {} != {}".format(
                    name,
                    features.shape[1],
                    skip.shape[1],
                )
            )
        if features.shape[-2:] != skip.shape[-2:]:
            features = F.interpolate(
                features,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return features

    @staticmethod
    def _cosine_map(first, second):
        first = F.normalize(first, p=2, dim=1, eps=1e-6)
        second = F.normalize(second, p=2, dim=1, eps=1e-6)
        return (first * second).sum(dim=1, keepdim=True)

    def _channel_gain(self, skip_low, plam_low, decoder_low, text):
        descriptor_source = skip_low + plam_low + decoder_low
        average_pool = descriptor_source.mean(dim=(2, 3))
        maximum_pool = descriptor_source.amax(dim=(2, 3))
        if self.normalize_channel_descriptors:
            descriptor_scale = math.sqrt(descriptor_source.shape[1])
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

    def _semantic_region(self, skip_low, plam_low, decoder_low, text):
        skip_features = self.skip_semantic_norm(
            self.skip_semantic_proj(skip_low)
        )
        plam_features = self.plam_semantic_norm(
            self.plam_semantic_proj(plam_low)
        )
        decoder_features = self.decoder_semantic_norm(
            self.decoder_semantic_proj(decoder_low)
        )
        if not self.use_plam_guide:
            plam_features = torch.zeros_like(plam_features)
        if not self.use_decoder_guide:
            decoder_features = torch.zeros_like(decoder_features)

        semantic_features = skip_features + plam_features + decoder_features
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
        semantic_average = semantic_features.mean(dim=1, keepdim=True)
        semantic_maximum = semantic_features.amax(dim=1, keepdim=True)
        plam_agreement = self._cosine_map(skip_features, plam_features)
        decoder_agreement = self._cosine_map(
            skip_features,
            decoder_features,
        )
        region_logit = self.region_spatial(
            torch.cat(
                [
                    semantic_average,
                    semantic_maximum,
                    plam_agreement,
                    decoder_agreement,
                ],
                dim=1,
            )
        )
        semantic_support = torch.sigmoid(region_logit)
        region_residual = (
            self.region_out(semantic_features) * semantic_support
        )
        return (
            region_residual,
            semantic_support,
            skip_features,
            plam_features,
            decoder_features,
            semantic_features,
            plam_agreement,
            decoder_agreement,
            text_film_magnitude,
        )

    def _detail_residual(self, skip_high, semantic_support):
        local_features = self.detail_local(skip_high)
        context_features = self.detail_context(skip_high)
        if not self.use_dilated_detail:
            context_features = torch.zeros_like(context_features)
        detail_features = F.silu(
            self.detail_norm(local_features + context_features)
        )
        detail_refinement = self.detail_out(detail_features)
        support_gain = 0.5 + semantic_support
        detail_residual = skip_high * support_gain + detail_refinement
        return (
            detail_residual,
            detail_refinement,
            local_features,
            context_features,
            detail_features,
        )

    def forward(self, skip, plam=None, decoder=None, text=None):
        plam = self._at_skip_resolution(plam, skip, "PLAM")
        decoder = self._at_skip_resolution(decoder, skip, "Decoder")

        skip_low, skip_high = self.frequency_split(skip)
        plam_low, plam_high = self.frequency_split(plam)
        decoder_low, _ = self.frequency_split(decoder)

        if not self.use_plam_guide:
            plam_low = torch.zeros_like(plam_low)
            plam_high = torch.zeros_like(plam_high)
        if not self.use_decoder_guide:
            decoder_low = torch.zeros_like(decoder_low)

        channel_gain = self._channel_gain(
            skip_low,
            plam_low,
            decoder_low,
            text,
        )
        (
            region_residual,
            semantic_support,
            skip_features,
            plam_features,
            decoder_features,
            semantic_features,
            plam_agreement,
            decoder_agreement,
            text_film_magnitude,
        ) = self._semantic_region(
            skip_low,
            plam_low,
            decoder_low,
            text,
        )
        (
            detail_residual,
            detail_refinement,
            local_features,
            context_features,
            detail_features,
        ) = self._detail_residual(skip_high, semantic_support)

        region_strength = (
            self.region_strength_max
            * torch.sigmoid(self.region_strength_logit)
        )
        detail_strength = (
            self.detail_strength_floor
            + (self.detail_strength_max - self.detail_strength_floor)
            * torch.sigmoid(self.detail_strength_logit)
        )
        plam_strength = (
            self.plam_strength_floor
            + (self.plam_strength_max - self.plam_strength_floor)
            * torch.sigmoid(self.plam_strength_logit)
        )

        channel_residual = skip_low * (channel_gain - 1.0)
        output = (
            skip
            + plam_strength * plam_low
            + channel_residual
            + region_strength * region_residual
            + detail_strength * detail_residual
        )

        if not self.training:
            with torch.no_grad():
                reconstruction_error = (
                    skip - (skip_low + skip_high)
                ).abs().amax()
                branch_energy = torch.stack(
                    [
                        skip_low.abs().mean(),
                        plam_low.abs().mean(),
                        decoder_low.abs().mean(),
                        skip_high.abs().mean(),
                    ]
                ).clamp_min(1e-8)
                branch_weights = branch_energy / branch_energy.sum()
                spatial_residual = (
                    region_strength
                    * torch.tanh(region_residual.mean(dim=1, keepdim=True))
                    + detail_strength
                    * torch.tanh(detail_residual.mean(dim=1, keepdim=True))
                )
                spatial_gain = 1.0 + spatial_residual
                skip_energy = skip.square().mean().clamp_min(1e-8)
                plam_energy = plam.square().mean().clamp_min(1e-8)
                self._last_stats = {
                    "architecture_version": self.architecture_version,
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
                            + detail_features.abs().mean()
                        ).item()
                    ),
                    "spatial_local_mean": float(
                        region_residual.mean().item()
                    ),
                    "spatial_global_mean": float(
                        detail_residual.mean().item()
                    ),
                    "local_strength_mean": float(
                        region_strength.item()
                    ),
                    "global_strength_mean": float(
                        detail_strength.item()
                    ),
                    "spatial_saturation_ratio": float(
                        semantic_support.lt(0.05).logical_or(
                            semantic_support.gt(0.95)
                        ).float().mean().item()
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
                    "guide_plam_weight": float(branch_weights[1].item()),
                    "guide_decoder_weight": float(branch_weights[2].item()),
                    "guide_detail_weight": float(branch_weights[3].item()),
                    "haar_reconstruction_error": float(
                        reconstruction_error.item()
                    ),
                    "skip_low_energy_ratio": float(
                        (skip_low.square().mean() / skip_energy).item()
                    ),
                    "skip_high_energy_ratio": float(
                        (skip_high.square().mean() / skip_energy).item()
                    ),
                    "plam_low_energy_ratio": float(
                        (plam_low.square().mean() / plam_energy).item()
                    ),
                    "plam_high_energy_ratio": float(
                        (plam_high.square().mean() / plam_energy).item()
                    ),
                    "plam_strength_mean": float(plam_strength.item()),
                    "region_strength_mean": float(region_strength.item()),
                    "detail_strength_mean": float(detail_strength.item()),
                    "region_residual_std": float(
                        region_residual.std().item()
                    ),
                    "detail_residual_std": float(
                        detail_residual.std().item()
                    ),
                    "detail_refinement_std": float(
                        detail_refinement.std().item()
                    ),
                    "semantic_support_mean": float(
                        semantic_support.mean().item()
                    ),
                    "semantic_support_std": float(
                        semantic_support.std().item()
                    ),
                    "plam_skip_agreement": float(
                        plam_agreement.mean().item()
                    ),
                    "decoder_skip_agreement": float(
                        decoder_agreement.mean().item()
                    ),
                    "raw_skip_feature_abs_mean": float(
                        skip_features.abs().mean().item()
                    ),
                    "plam_feature_abs_mean": float(
                        plam_features.abs().mean().item()
                    ),
                    "decoder_feature_abs_mean": float(
                        decoder_features.abs().mean().item()
                    ),
                    "detail_local_abs_mean": float(
                        local_features.abs().mean().item()
                    ),
                    "detail_context_abs_mean": float(
                        context_features.abs().mean().item()
                    ),
                }
        return output


# Keep the public name used by the existing LViT import path.
EPPA = FAMHaarEPPA

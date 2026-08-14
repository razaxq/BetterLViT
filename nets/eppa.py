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


class SpatialAdaptiveFrequencyRefiner(nn.Module):
    """Efficient adaptive frequency filtering and optional flow alignment.

    FreqFusion predicts a normalized filter for every spatial location. A
    literal CARAFE/unfold port is unnecessarily expensive for this 24 GB AMD
    workstation, so V4-B predicts mixtures over a fixed normalized filter
    bank: identity, 3x3 binomial and 5x5 binomial. The mixture remains
    spatially variant and group-specific while every candidate filter is
    implemented by an efficient depthwise convolution.

    V4-C adds the offset component omitted by V4-B. It predicts bounded,
    group-wise semantic flow from the shared context and its eight-neighbour
    cosine similarities, then warps the decoder with ``grid_sample``. The
    final flow predictor is zero-initialized, so V4-C is exactly V4-B at
    initialization. The ALPF path smooths the aligned decoder feature before
    semantic fusion.
    The AHPF path adds the complementary high-frequency skip residual. Both
    paths have independent bounded strengths and therefore cannot silently
    become an unconstrained replacement for the V4-A residual backbone.
    """

    filter_names = ("identity", "blur3", "blur5")

    def __init__(
        self,
        channels,
        groups=8,
        context_channels=32,
        alpf_strength_max=0.50,
        alpf_strength_init=0.20,
        ahpf_strength_max=0.30,
        ahpf_strength_init=0.08,
        ahpf_strength_floor=0.02,
        use_semantic_flow_alignment=False,
        flow_groups=4,
        flow_max_offset=1.5,
        flow_strength_max=1.0,
        flow_strength_init=0.25,
    ):
        super().__init__()
        if channels % groups:
            raise ValueError(
                "Adaptive frequency groups must divide channels: {} % {}"
                .format(channels, groups)
            )
        self.channels = int(channels)
        self.groups = int(groups)
        self.context_channels = int(context_channels)
        self.alpf_strength_max = float(alpf_strength_max)
        self.ahpf_strength_max = float(ahpf_strength_max)
        self.ahpf_strength_floor = float(ahpf_strength_floor)
        self.use_semantic_flow_alignment = bool(
            use_semantic_flow_alignment
        )
        self.flow_groups = int(flow_groups)
        self.flow_max_offset = float(flow_max_offset)
        self.flow_strength_max = float(flow_strength_max)
        if self.use_semantic_flow_alignment and channels % self.flow_groups:
            raise ValueError(
                "Semantic-flow groups must divide channels: {} % {}".format(
                    channels,
                    self.flow_groups,
                )
            )

        self.skip_context = nn.Conv2d(
            channels,
            context_channels,
            kernel_size=1,
            bias=False,
        )
        self.plam_context = nn.Conv2d(
            channels,
            context_channels,
            kernel_size=1,
            bias=False,
        )
        self.decoder_context = nn.Conv2d(
            channels,
            context_channels,
            kernel_size=1,
            bias=False,
        )
        context_groups = min(8, context_channels)
        while context_channels % context_groups:
            context_groups -= 1
        self.context_norm = nn.GroupNorm(
            context_groups,
            context_channels,
        )
        filter_count = len(self.filter_names)
        output_channels = groups * filter_count
        self.low_kernel_predictor = nn.Conv2d(
            context_channels,
            output_channels,
            kernel_size=3,
            padding=1,
        )
        self.high_kernel_predictor = nn.Conv2d(
            context_channels,
            output_channels,
            kernel_size=3,
            padding=1,
        )
        nn.init.normal_(self.low_kernel_predictor.weight, std=0.001)
        nn.init.normal_(self.high_kernel_predictor.weight, std=0.001)
        low_bias = torch.tensor([0.0, 0.75, -0.75]).repeat(groups)
        high_bias = torch.tensor([0.0, 0.50, -0.50]).repeat(groups)
        with torch.no_grad():
            self.low_kernel_predictor.bias.copy_(low_bias)
            self.high_kernel_predictor.bias.copy_(high_bias)

        if self.use_semantic_flow_alignment:
            flow_hidden = max(context_channels, 16)
            self.flow_predictor = nn.Sequential(
                nn.Conv2d(
                    context_channels + 8,
                    flow_hidden,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, flow_hidden),
                nn.SiLU(inplace=True),
                nn.Conv2d(
                    flow_hidden,
                    self.flow_groups * 2,
                    kernel_size=3,
                    padding=1,
                ),
            )
            nn.init.zeros_(self.flow_predictor[-1].weight)
            nn.init.zeros_(self.flow_predictor[-1].bias)
            self.flow_strength_logit = nn.Parameter(
                torch.tensor(
                    _strength_logit(
                        flow_strength_init,
                        self.flow_strength_max,
                    )
                ).view(1, 1, 1, 1)
            )

        blur3_vector = torch.tensor([1.0, 2.0, 1.0])
        blur3 = torch.outer(blur3_vector, blur3_vector)
        blur3 = blur3 / blur3.sum()
        blur5_vector = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        blur5 = torch.outer(blur5_vector, blur5_vector)
        blur5 = blur5 / blur5.sum()
        self.register_buffer("blur3_kernel", blur3[None, None])
        self.register_buffer("blur5_kernel", blur5[None, None])

        self.alpf_strength_logit = nn.Parameter(
            torch.tensor(
                _strength_logit(
                    alpf_strength_init,
                    self.alpf_strength_max,
                )
            ).view(1, 1, 1, 1)
        )
        self.ahpf_strength_logit = nn.Parameter(
            torch.tensor(
                _strength_logit(
                    ahpf_strength_init,
                    self.ahpf_strength_max,
                    self.ahpf_strength_floor,
                )
            ).view(1, 1, 1, 1)
        )

    def _depthwise_filter(self, features, kernel):
        kernel_size = kernel.shape[-1]
        padding = kernel_size // 2
        padded = F.pad(
            features,
            (padding, padding, padding, padding),
            mode="reflect",
        )
        weights = kernel.to(dtype=features.dtype).expand(
            self.channels,
            1,
            kernel_size,
            kernel_size,
        )
        return F.conv2d(padded, weights, groups=self.channels)

    def _mixture_weights(self, logits):
        batch, _, height, width = logits.shape
        return F.softmax(
            logits.view(
                batch,
                self.groups,
                len(self.filter_names),
                height,
                width,
            ),
            dim=2,
        )

    def _apply_filter_bank(self, features, mixture_weights):
        responses = (
            features,
            self._depthwise_filter(features, self.blur3_kernel),
            self._depthwise_filter(features, self.blur5_kernel),
        )
        output = torch.zeros_like(features)
        for index, response in enumerate(responses):
            channel_weights = mixture_weights[:, :, index]
            channel_weights = channel_weights.repeat_interleave(
                self.channels // self.groups,
                dim=1,
            )
            output = output + response * channel_weights
        return output

    @staticmethod
    def _local_similarity(features):
        """Return cosine similarity to the eight immediate neighbours."""
        normalized = F.normalize(features, p=2, dim=1, eps=1e-6)
        height, width = normalized.shape[-2:]
        padded = F.pad(normalized, (1, 1, 1, 1), mode="replicate")
        similarities = []
        for offset_y, offset_x in (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ):
            neighbour = padded[
                :,
                :,
                1 + offset_y:1 + offset_y + height,
                1 + offset_x:1 + offset_x + width,
            ]
            similarities.append(
                (normalized * neighbour).sum(dim=1, keepdim=True)
            )
        return torch.cat(similarities, dim=1)

    def _warp_groups(self, features, flow):
        """Warp channel groups using pixel-unit x/y semantic flow."""
        batch, channels, height, width = features.shape
        channels_per_group = channels // self.flow_groups
        if flow.shape != (batch, self.flow_groups, 2, height, width):
            raise ValueError(
                "Unexpected semantic-flow shape: {}".format(
                    tuple(flow.shape)
                )
            )

        y_coordinates = torch.linspace(
            -1.0,
            1.0,
            height,
            device=features.device,
            dtype=features.dtype,
        )
        x_coordinates = torch.linspace(
            -1.0,
            1.0,
            width,
            device=features.device,
            dtype=features.dtype,
        )
        grid_y, grid_x = torch.meshgrid(
            y_coordinates,
            x_coordinates,
            indexing="ij",
        )
        base_grid = torch.stack((grid_x, grid_y), dim=-1)

        normalized_flow = torch.stack(
            (
                flow[:, :, 0] * (2.0 / max(width - 1, 1)),
                flow[:, :, 1] * (2.0 / max(height - 1, 1)),
            ),
            dim=-1,
        )
        sampling_grid = (
            base_grid[None, None]
            + normalized_flow
        ).reshape(batch * self.flow_groups, height, width, 2)
        grouped_features = features.reshape(
            batch * self.flow_groups,
            channels_per_group,
            height,
            width,
        )
        aligned = F.grid_sample(
            grouped_features,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return aligned.reshape(batch, channels, height, width)

    def _align_decoder(self, context, decoder):
        local_similarity = self._local_similarity(context)
        raw_flow = self.flow_predictor(
            torch.cat((context, local_similarity), dim=1)
        )
        batch, _, height, width = raw_flow.shape
        raw_flow = raw_flow.view(
            batch,
            self.flow_groups,
            2,
            height,
            width,
        )
        flow_strength = (
            self.flow_strength_max
            * torch.sigmoid(self.flow_strength_logit)
        )
        flow_direction = torch.tanh(raw_flow)
        flow_norm = flow_direction.square().sum(
            dim=2,
            keepdim=True,
        ).add(1e-6).sqrt()
        flow_direction = flow_direction / flow_norm.clamp_min(1.0)
        effective_flow = (
            self.flow_max_offset * flow_strength * flow_direction
        )
        aligned_decoder = self._warp_groups(decoder, effective_flow)
        return (
            aligned_decoder,
            effective_flow,
            flow_strength,
            local_similarity,
        )

    def forward(
        self,
        skip_low,
        plam_low,
        decoder_low,
        skip,
        decoder,
    ):
        context = self.context_norm(
            self.skip_context(skip_low)
            + self.plam_context(plam_low)
            + self.decoder_context(decoder_low)
        )
        context = F.silu(context)
        low_weights = self._mixture_weights(
            self.low_kernel_predictor(context)
        )
        high_weights = self._mixture_weights(
            self.high_kernel_predictor(context)
        )

        decoder_aligned = decoder
        effective_flow = None
        flow_strength = None
        local_similarity = None
        if self.use_semantic_flow_alignment:
            (
                decoder_aligned,
                effective_flow,
                flow_strength,
                local_similarity,
            ) = self._align_decoder(context, decoder)

        decoder_filtered = self._apply_filter_bank(
            decoder_aligned,
            low_weights,
        )
        skip_filtered = self._apply_filter_bank(skip, high_weights)
        decoder_delta = decoder_filtered - decoder_aligned
        alignment_delta = decoder_aligned - decoder
        skip_high_residual = skip - skip_filtered

        alpf_strength = (
            self.alpf_strength_max
            * torch.sigmoid(self.alpf_strength_logit)
        )
        ahpf_strength = (
            self.ahpf_strength_floor
            + (self.ahpf_strength_max - self.ahpf_strength_floor)
            * torch.sigmoid(self.ahpf_strength_logit)
        )
        decoder_adaptive = decoder_aligned + alpf_strength * decoder_delta
        skip_adaptive_residual = ahpf_strength * skip_high_residual
        diagnostics = {
            "low_weights": low_weights,
            "high_weights": high_weights,
            "decoder_delta": decoder_delta,
            "skip_high_residual": skip_high_residual,
            "skip_adaptive_residual": skip_adaptive_residual,
            "alpf_strength": alpf_strength,
            "ahpf_strength": ahpf_strength,
            "semantic_flow_enabled": bool(
                self.use_semantic_flow_alignment
            ),
            "decoder_aligned": decoder_aligned,
            "alignment_delta": alignment_delta,
            "effective_flow": effective_flow,
            "flow_strength": flow_strength,
            "local_similarity": local_similarity,
        }
        return decoder_adaptive, skip_adaptive_residual, diagnostics


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
        use_adaptive_frequency=False,
        frequency_groups=8,
        frequency_context_channels=32,
        alpf_strength_max=0.50,
        alpf_strength_init=0.20,
        ahpf_strength_max=0.30,
        ahpf_strength_init=0.08,
        ahpf_strength_floor=0.02,
        use_semantic_flow_alignment=False,
        flow_groups=4,
        flow_max_offset=1.5,
        flow_strength_max=1.0,
        flow_strength_init=0.25,
        use_token_routing=False,
        token_attention_dim=32,
        token_attention_heads=4,
        token_strength_max=0.50,
        token_strength_init=0.10,
        token_temperature_init=5.0,
        use_plam_calibration=False,
        plam_calibration_max_delta=0.50,
        plam_calibration_hidden_channels=16,
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
        self.use_adaptive_frequency = bool(use_adaptive_frequency)
        self.use_semantic_flow_alignment = bool(
            use_semantic_flow_alignment
        )
        self.use_token_routing = bool(use_token_routing)
        self.use_plam_calibration = bool(use_plam_calibration)
        self.plam_calibration_max_delta = float(
            plam_calibration_max_delta
        )
        if self.use_semantic_flow_alignment and not self.use_adaptive_frequency:
            raise ValueError(
                "Semantic-flow alignment requires adaptive frequency routing"
            )
        if self.use_plam_calibration and not self.use_plam_guide:
            raise ValueError("PLAM calibration requires the PLAM guide")
        if not 0.0 < self.plam_calibration_max_delta < 1.0:
            raise ValueError(
                "PLAM calibration delta must be between zero and one"
            )

        self.frequency_split = FixedHaarFrequencySplit(in_channels)
        if self.use_adaptive_frequency:
            self.adaptive_frequency = SpatialAdaptiveFrequencyRefiner(
                in_channels,
                groups=frequency_groups,
                context_channels=frequency_context_channels,
                alpf_strength_max=alpf_strength_max,
                alpf_strength_init=alpf_strength_init,
                ahpf_strength_max=ahpf_strength_max,
                ahpf_strength_init=ahpf_strength_init,
                ahpf_strength_floor=ahpf_strength_floor,
                use_semantic_flow_alignment=(
                    self.use_semantic_flow_alignment
                ),
                flow_groups=flow_groups,
                flow_max_offset=flow_max_offset,
                flow_strength_max=flow_strength_max,
                flow_strength_init=flow_strength_init,
            )

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

        if self.use_plam_calibration:
            calibration_hidden = max(
                4,
                int(plam_calibration_hidden_channels),
            )
            self.plam_calibrator = nn.Sequential(
                nn.Conv2d(
                    4,
                    calibration_hidden,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, calibration_hidden),
                nn.SiLU(inplace=True),
                nn.Conv2d(
                    calibration_hidden,
                    1,
                    kernel_size=1,
                ),
            )
            # V4-E starts as an exact V4-B function. Training can then learn
            # only a bounded, spatially varying correction to the PLAM path.
            nn.init.zeros_(self.plam_calibrator[-1].weight)
            nn.init.zeros_(self.plam_calibrator[-1].bias)

        if self.use_token_routing:
            if text_dim is None:
                raise ValueError("Token routing requires text embeddings")
            if token_attention_dim % token_attention_heads:
                raise ValueError(
                    "Token attention heads must divide attention channels: "
                    "{} % {}".format(
                        token_attention_dim,
                        token_attention_heads,
                    )
                )
            self.token_attention_dim = int(token_attention_dim)
            self.token_attention_heads = int(token_attention_heads)
            self.token_head_dim = (
                self.token_attention_dim // self.token_attention_heads
            )
            self.token_strength_max = float(token_strength_max)
            self.token_query = nn.Conv2d(
                guide_channels,
                self.token_attention_dim,
                kernel_size=1,
                bias=False,
            )
            self.token_key = nn.Linear(
                text_dim,
                self.token_attention_dim,
                bias=False,
            )
            self.token_value = nn.Linear(
                text_dim,
                self.token_attention_dim,
                bias=False,
            )
            self.token_out = nn.Conv2d(
                self.token_attention_dim,
                guide_channels,
                kernel_size=1,
                bias=False,
            )
            # Safe initialization: V4-D exactly reproduces V4-B before the
            # token-localized residual starts learning.
            nn.init.zeros_(self.token_out.weight)
            self.token_strength_logit = nn.Parameter(
                torch.tensor(
                    _strength_logit(
                        token_strength_init,
                        self.token_strength_max,
                    )
                )
            )
            self.token_log_temperature = nn.Parameter(
                torch.tensor(math.log(token_temperature_init))
            )

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

    def _token_localize(self, visual_features, text, text_mask):
        if text is None:
            raise ValueError("Token routing requires a text tensor")
        if text.ndim != 3 or text.shape[2] != self.text_dim:
            raise ValueError(
                "Unexpected text shape for token routing: {}".format(
                    tuple(text.shape)
                )
            )
        batch, _, height, width = visual_features.shape
        if text.shape[0] != batch:
            raise ValueError("Text and visual batch sizes must match")
        if text_mask is None:
            text_mask = torch.ones(
                text.shape[:2],
                dtype=torch.bool,
                device=text.device,
            )
        else:
            if tuple(text_mask.shape) != tuple(text.shape[:2]):
                raise ValueError(
                    "Unexpected text mask shape: {}".format(
                        tuple(text_mask.shape)
                    )
                )
            text_mask = text_mask.to(device=text.device, dtype=torch.bool)
        if not text_mask.any(dim=1).all():
            raise ValueError("Every sample must contain at least one text token")

        query = self.token_query(visual_features).reshape(
            batch,
            self.token_attention_heads,
            self.token_head_dim,
            height * width,
        ).permute(0, 1, 3, 2)
        key = self.token_key(text).reshape(
            batch,
            text.shape[1],
            self.token_attention_heads,
            self.token_head_dim,
        ).permute(0, 2, 1, 3)
        value = self.token_value(text).reshape(
            batch,
            text.shape[1],
            self.token_attention_heads,
            self.token_head_dim,
        ).permute(0, 2, 1, 3)
        query = F.normalize(query, p=2, dim=-1, eps=1e-6)
        key = F.normalize(key, p=2, dim=-1, eps=1e-6)
        temperature = self.token_log_temperature.clamp(
            min=math.log(1.0),
            max=math.log(20.0),
        ).exp()
        attention_logits = torch.einsum(
            "bhnd,bhtd->bhnt",
            query,
            key,
        ) * temperature
        attention_logits = attention_logits.masked_fill(
            ~text_mask[:, None, None, :],
            torch.finfo(attention_logits.dtype).min,
        )
        attention = attention_logits.softmax(dim=-1)
        attended_text = torch.einsum(
            "bhnt,bhtd->bhnd",
            attention,
            value,
        ).permute(0, 1, 3, 2).reshape(
            batch,
            self.token_attention_dim,
            height,
            width,
        )
        token_residual = self.token_out(attended_text)
        token_strength = (
            self.token_strength_max
            * torch.sigmoid(self.token_strength_logit)
        )
        routed_features = visual_features + token_strength * token_residual
        diagnostics = {
            "attention": attention,
            "text_mask": text_mask,
            "token_residual": token_residual,
            "token_strength": token_strength,
            "temperature": temperature,
        }
        return routed_features, diagnostics

    def _semantic_region(
        self,
        skip_low,
        plam_low,
        decoder_low,
        text,
        text_mask,
    ):
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

        plam_agreement = self._cosine_map(skip_features, plam_features)
        decoder_agreement = self._cosine_map(
            skip_features,
            decoder_features,
        )
        plam_decoder_agreement = self._cosine_map(
            plam_features,
            decoder_features,
        )
        plam_calibration_gate = torch.ones_like(plam_agreement)
        if self.use_plam_calibration:
            calibration_evidence = torch.cat(
                [
                    plam_agreement,
                    decoder_agreement,
                    plam_decoder_agreement,
                    (plam_agreement - decoder_agreement).abs(),
                ],
                dim=1,
            )
            calibration_delta = self.plam_calibrator(
                calibration_evidence
            )
            plam_calibration_gate = (
                1.0
                + self.plam_calibration_max_delta
                * torch.tanh(calibration_delta)
            )

        calibrated_plam_features = (
            plam_features * plam_calibration_gate
        )
        semantic_features = (
            skip_features
            + calibrated_plam_features
            + decoder_features
        )
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

        token_diagnostics = None
        if self.use_token_routing:
            semantic_features, token_diagnostics = self._token_localize(
                semantic_features,
                text,
                text_mask,
            )

        semantic_features = F.silu(semantic_features)
        semantic_average = semantic_features.mean(dim=1, keepdim=True)
        semantic_maximum = semantic_features.amax(dim=1, keepdim=True)
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
            token_diagnostics,
            plam_calibration_gate,
            plam_decoder_agreement,
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

    def forward(
        self,
        skip,
        plam=None,
        decoder=None,
        text=None,
        text_mask=None,
        return_decoder=False,
    ):
        plam = self._at_skip_resolution(plam, skip, "PLAM")
        decoder = self._at_skip_resolution(decoder, skip, "Decoder")
        decoder_for_fusion = decoder

        skip_low, skip_high = self.frequency_split(skip)
        plam_low, plam_high = self.frequency_split(plam)
        decoder_low, _ = self.frequency_split(decoder)

        if not self.use_plam_guide:
            plam_low = torch.zeros_like(plam_low)
            plam_high = torch.zeros_like(plam_high)
        if not self.use_decoder_guide:
            decoder_low = torch.zeros_like(decoder_low)

        adaptive_diagnostics = None
        adaptive_skip_residual = torch.zeros_like(skip)
        if self.use_adaptive_frequency:
            decoder_source = (
                decoder
                if self.use_decoder_guide
                else torch.zeros_like(decoder)
            )
            (
                decoder_adaptive,
                adaptive_skip_residual,
                adaptive_diagnostics,
            ) = self.adaptive_frequency(
                skip_low,
                plam_low,
                decoder_low,
                skip,
                decoder_source,
            )
            decoder_for_fusion = decoder_adaptive
            decoder_low, _ = self.frequency_split(decoder_adaptive)
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
            token_diagnostics,
            plam_calibration_gate,
            plam_decoder_agreement,
        ) = self._semantic_region(
            skip_low,
            plam_low,
            decoder_low,
            text,
            text_mask,
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
            + adaptive_skip_residual
            + plam_strength * plam_low * plam_calibration_gate
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
                    torch.tanh(
                        adaptive_skip_residual.mean(dim=1, keepdim=True)
                    )
                    + region_strength
                    * torch.tanh(region_residual.mean(dim=1, keepdim=True))
                    + detail_strength
                    * torch.tanh(detail_residual.mean(dim=1, keepdim=True))
                )
                spatial_gain = 1.0 + spatial_residual
                skip_energy = skip.square().mean().clamp_min(1e-8)
                plam_energy = plam.square().mean().clamp_min(1e-8)
                self._last_stats = {
                    "architecture_version": self.architecture_version,
                    "adaptive_frequency_enabled": bool(
                        self.use_adaptive_frequency
                    ),
                    "token_routing_enabled": bool(
                        self.use_token_routing
                    ),
                    "plam_calibration_enabled": bool(
                        self.use_plam_calibration
                    ),
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
                    "plam_decoder_agreement": float(
                        plam_decoder_agreement.mean().item()
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
                if self.use_plam_calibration:
                    flat_gate = plam_calibration_gate.flatten()
                    flat_agreement = plam_agreement.flatten()
                    centered_gate = flat_gate - flat_gate.mean()
                    centered_agreement = (
                        flat_agreement - flat_agreement.mean()
                    )
                    gate_agreement_correlation = (
                        centered_gate.mul(centered_agreement).mean()
                        / (
                            centered_gate.square().mean().sqrt()
                            * centered_agreement.square().mean().sqrt()
                        ).clamp_min(1e-8)
                    )
                    calibrated_plam = plam_low * plam_calibration_gate
                    self._last_stats.update({
                        "plam_calibration_gate_mean": float(
                            plam_calibration_gate.mean().item()
                        ),
                        "plam_calibration_gate_std": float(
                            plam_calibration_gate.std().item()
                        ),
                        "plam_calibration_gate_min": float(
                            plam_calibration_gate.amin().item()
                        ),
                        "plam_calibration_gate_max": float(
                            plam_calibration_gate.amax().item()
                        ),
                        "plam_calibration_amplify_ratio": float(
                            (plam_calibration_gate > 1.05)
                            .float()
                            .mean()
                            .item()
                        ),
                        "plam_calibration_suppress_ratio": float(
                            (plam_calibration_gate < 0.95)
                            .float()
                            .mean()
                            .item()
                        ),
                        "plam_calibration_agreement_correlation": float(
                            gate_agreement_correlation.item()
                        ),
                        "plam_calibrated_residual_std": float(
                            calibrated_plam.std().item()
                        ),
                    })
                if token_diagnostics is not None:
                    attention = token_diagnostics["attention"]
                    token_mask = token_diagnostics["text_mask"]
                    valid_count = token_mask.sum(dim=1).float()
                    attention_entropy = (
                        -attention
                        * attention.clamp_min(1e-8).log()
                    ).sum(dim=-1)
                    attention_entropy = (
                        attention_entropy
                        / valid_count.clamp_min(2.0).log()[:, None, None]
                    )
                    spatial_attention = attention.mean(dim=1)
                    spatial_std = spatial_attention.std(dim=1)
                    masked_spatial_std = (
                        spatial_std * token_mask.float()
                    ).sum() / token_mask.sum().clamp_min(1)
                    self._last_stats.update({
                        "token_strength_mean": float(
                            token_diagnostics["token_strength"].item()
                        ),
                        "token_temperature": float(
                            token_diagnostics["temperature"].item()
                        ),
                        "token_attention_entropy": float(
                            attention_entropy.mean().item()
                        ),
                        "token_attention_peak": float(
                            attention.amax(dim=-1).mean().item()
                        ),
                        "token_cls_mass": float(
                            attention[..., 0].mean().item()
                        ),
                        "token_non_cls_mass": float(
                            (1.0 - attention[..., 0]).mean().item()
                        ),
                        "token_attention_spatial_std": float(
                            masked_spatial_std.item()
                        ),
                        "token_residual_std": float(
                            token_diagnostics["token_residual"].std().item()
                        ),
                        "token_valid_count_mean": float(
                            valid_count.mean().item()
                        ),
                    })
                if adaptive_diagnostics is not None:
                    low_weights = adaptive_diagnostics["low_weights"]
                    high_weights = adaptive_diagnostics["high_weights"]
                    low_means = low_weights.mean(dim=(0, 1, 3, 4))
                    high_means = high_weights.mean(dim=(0, 1, 3, 4))
                    low_entropy = (
                        -low_weights
                        * low_weights.clamp_min(1e-8).log()
                    ).sum(dim=2).mean().div(
                        math.log(len(self.adaptive_frequency.filter_names))
                    )
                    high_entropy = (
                        -high_weights
                        * high_weights.clamp_min(1e-8).log()
                    ).sum(dim=2).mean().div(
                        math.log(len(self.adaptive_frequency.filter_names))
                    )
                    self._last_stats.update({
                        "alpf_strength_mean": float(
                            adaptive_diagnostics["alpf_strength"].item()
                        ),
                        "ahpf_strength_mean": float(
                            adaptive_diagnostics["ahpf_strength"].item()
                        ),
                        "alpf_kernel_sum": float(
                            low_weights.sum(dim=2).mean().item()
                        ),
                        "ahpf_kernel_sum": float(
                            high_weights.sum(dim=2).mean().item()
                        ),
                        "alpf_kernel_entropy": float(low_entropy.item()),
                        "ahpf_kernel_entropy": float(high_entropy.item()),
                        "alpf_identity_weight": float(low_means[0].item()),
                        "alpf_blur3_weight": float(low_means[1].item()),
                        "alpf_blur5_weight": float(low_means[2].item()),
                        "ahpf_identity_weight": float(high_means[0].item()),
                        "ahpf_blur3_weight": float(high_means[1].item()),
                        "ahpf_blur5_weight": float(high_means[2].item()),
                        "alpf_delta_std": float(
                            adaptive_diagnostics["decoder_delta"].std().item()
                        ),
                        "ahpf_residual_std": float(
                            adaptive_diagnostics[
                                "skip_high_residual"
                            ].std().item()
                        ),
                        "adaptive_skip_residual_std": float(
                            adaptive_diagnostics[
                                "skip_adaptive_residual"
                            ].std().item()
                        ),
                        "semantic_flow_enabled": bool(
                            adaptive_diagnostics[
                                "semantic_flow_enabled"
                            ]
                        ),
                    })
                    if adaptive_diagnostics["semantic_flow_enabled"]:
                        effective_flow = adaptive_diagnostics[
                            "effective_flow"
                        ]
                        flow_magnitude = effective_flow.square().sum(
                            dim=2
                        ).sqrt()
                        decoder_aligned = adaptive_diagnostics[
                            "decoder_aligned"
                        ]
                        agreement_before = self._cosine_map(
                            skip_low,
                            decoder,
                        )
                        agreement_after = self._cosine_map(
                            skip_low,
                            decoder_aligned,
                        )
                        local_similarity = adaptive_diagnostics[
                            "local_similarity"
                        ]
                        self._last_stats.update({
                            "flow_strength_mean": float(
                                adaptive_diagnostics[
                                    "flow_strength"
                                ].item()
                            ),
                            "flow_offset_mean": float(
                                flow_magnitude.mean().item()
                            ),
                            "flow_offset_max": float(
                                flow_magnitude.amax().item()
                            ),
                            "flow_active_ratio": float(
                                (flow_magnitude > 0.25)
                                .float()
                                .mean()
                                .item()
                            ),
                            "flow_alignment_delta_std": float(
                                adaptive_diagnostics[
                                    "alignment_delta"
                                ].std().item()
                            ),
                            "flow_skip_agreement_before": float(
                                agreement_before.mean().item()
                            ),
                            "flow_skip_agreement_after": float(
                                agreement_after.mean().item()
                            ),
                            "flow_local_similarity_mean": float(
                                local_similarity.mean().item()
                            ),
                            "flow_local_similarity_std": float(
                                local_similarity.std().item()
                            ),
                        })
        if return_decoder:
            return output, decoder_for_fusion
        return output


class FAMAdaptiveHaarEPPA(FAMHaarEPPA):
    """V4-B: V4-A plus adaptive ALPF/AHPF at selected stages."""

    architecture_version = "fam_eppa_v4b"


class FAMSemanticFlowEPPA(FAMHaarEPPA):
    """V4-C: V4-B plus local-similarity semantic-flow alignment."""

    architecture_version = "fam_eppa_v4c"


class FAMTokenLocalizedEPPA(FAMHaarEPPA):
    """V4-D: V4-B plus token-localized semantic routing."""

    architecture_version = "fam_eppa_v4d"


class FAMReliabilityCalibratedEPPA(FAMHaarEPPA):
    """V4-E: V4-B plus bounded deep PLAM reliability calibration."""

    architecture_version = "fam_eppa_v4e"


# Keep the public name used by the existing LViT import path.
EPPA = FAMReliabilityCalibratedEPPA

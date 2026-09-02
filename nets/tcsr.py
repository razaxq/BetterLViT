"""Text-conditioned routing across the four LViT encoder skip scales.

TCSR is deliberately independent from decoder fusion.  It observes all raw
CNN skip tensors and the full clinical-text token sequence, allocates a shared
routing budget across scales, and applies a text-conditioned spatial/channel
residual to each skip before PLAM or EPPA sees it.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextConditionedCrossScaleSkipRouter(nn.Module):
    """Route encoder skips with token, scale, spatial and channel context.

    The learnable residual gate is exactly zero at construction, so enabling
    the module preserves the unmodified LViT forward pass until optimization
    starts opening the route.  No adaptive pooling is used because the formal
    protocol requires deterministic CUDA backward execution.
    """

    architecture_version = "tcsr_v1"

    def __init__(
        self,
        skip_channels,
        text_dim=768,
        routing_dim=32,
        max_residual_strength=1.0,
    ):
        super().__init__()
        self.skip_channels = tuple(int(value) for value in skip_channels)
        self.num_scales = len(self.skip_channels)
        self.text_dim = int(text_dim)
        self.routing_dim = int(routing_dim)
        self.max_residual_strength = float(max_residual_strength)
        if self.num_scales < 2:
            raise ValueError("TCSR requires at least two skip scales.")
        if self.routing_dim <= 0:
            raise ValueError("routing_dim must be positive.")
        if self.max_residual_strength <= 0:
            raise ValueError("max_residual_strength must be positive.")

        self.visual_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, self.routing_dim, kernel_size=1),
                nn.GroupNorm(1, self.routing_dim),
                nn.SiLU(),
            )
            for channels in self.skip_channels
        ])
        self.visual_norms = nn.ModuleList([
            nn.LayerNorm(self.routing_dim)
            for _ in self.skip_channels
        ])
        self.text_key = nn.Linear(self.text_dim, self.routing_dim)
        self.text_value = nn.Linear(self.text_dim, self.routing_dim)
        self.text_norm = nn.LayerNorm(self.routing_dim)

        score_width = max(16, self.routing_dim)
        self.scale_score_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.routing_dim * 3, score_width),
                nn.SiLU(),
                nn.Linear(score_width, 1),
            )
            for _ in self.skip_channels
        ])
        for head in self.scale_score_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

        self.context_fusion = nn.Sequential(
            nn.Linear(self.routing_dim * 2, self.routing_dim),
            nn.SiLU(),
            nn.LayerNorm(self.routing_dim),
        )
        self.channel_heads = nn.ModuleList([
            nn.Linear(self.routing_dim * 2, channels)
            for channels in self.skip_channels
        ])
        self.spatial_local = nn.ModuleList([
            nn.Conv2d(self.routing_dim, 1, kernel_size=1)
            for _ in self.skip_channels
        ])
        self.spatial_text = nn.ModuleList([
            nn.Linear(self.routing_dim, 1)
            for _ in self.skip_channels
        ])
        self.spatial_context = nn.ModuleList([
            nn.Linear(self.routing_dim, 1)
            for _ in self.skip_channels
        ])

        # tanh bounds the effective residual while preserving an exact zero
        # identity initialization and a non-zero gradient for each gate.
        self.residual_gate = nn.Parameter(torch.zeros(self.num_scales))
        self._last_stats = None

    @staticmethod
    def _validated_text_mask(text, text_mask):
        if text_mask is None:
            return torch.ones(
                text.shape[:2],
                dtype=torch.bool,
                device=text.device,
            )
        if tuple(text_mask.shape) != tuple(text.shape[:2]):
            raise ValueError(
                "text_mask shape {} does not match text tokens {}.".format(
                    tuple(text_mask.shape),
                    tuple(text.shape[:2]),
                )
            )
        mask = text_mask.to(device=text.device, dtype=torch.bool)
        # A fully padded row is invalid input, but keeping the first token makes
        # the router numerically safe and mirrors the mandatory BERT CLS token.
        empty_rows = ~mask.any(dim=1)
        if empty_rows.any():
            mask = mask.clone()
            mask[empty_rows, 0] = True
        return mask

    def _attend_text(self, descriptor, text_keys, text_values, text_mask):
        logits = torch.einsum("bd,bld->bl", descriptor, text_keys)
        logits = logits / math.sqrt(self.routing_dim)
        logits = logits.masked_fill(~text_mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
        attended = torch.einsum("bl,bld->bd", weights, text_values)
        return self.text_norm(attended), weights

    def forward(self, skips, text, text_mask=None):
        if len(skips) != self.num_scales:
            raise ValueError(
                "Expected {} skip scales, received {}.".format(
                    self.num_scales,
                    len(skips),
                )
            )
        if text.ndim != 3 or text.shape[-1] != self.text_dim:
            raise ValueError(
                "Expected text [B, L, {}], received {}.".format(
                    self.text_dim,
                    tuple(text.shape),
                )
            )
        batch_size = text.shape[0]
        projected = []
        descriptors = []
        for index, (skip, channels, projection, norm) in enumerate(zip(
            skips,
            self.skip_channels,
            self.visual_projections,
            self.visual_norms,
        )):
            if skip.ndim != 4:
                raise ValueError(
                    "Skip {} must be BCHW, received {}.".format(
                        index,
                        tuple(skip.shape),
                    )
                )
            if skip.shape[0] != batch_size or skip.shape[1] != channels:
                raise ValueError(
                    "Skip {} expected [B, {}, H, W], received {}.".format(
                        index,
                        channels,
                        tuple(skip.shape),
                    )
                )
            feature = projection(skip)
            projected.append(feature)
            descriptors.append(norm(feature.mean(dim=(2, 3))))

        mask = self._validated_text_mask(text, text_mask)
        text_keys = self.text_key(text)
        text_values = self.text_value(text)
        attended_text = []
        token_weights = []
        scale_logits = []
        for descriptor, score_head in zip(
            descriptors,
            self.scale_score_heads,
        ):
            attended, weights = self._attend_text(
                descriptor,
                text_keys,
                text_values,
                mask,
            )
            attended_text.append(attended)
            token_weights.append(weights)
            score_input = torch.cat(
                (descriptor, attended, descriptor * attended),
                dim=1,
            )
            scale_logits.append(score_head(score_input))

        scale_weights = torch.softmax(torch.cat(scale_logits, dim=1), dim=1)
        scale_contexts = [
            self.context_fusion(torch.cat((descriptor, attended), dim=1))
            for descriptor, attended in zip(descriptors, attended_text)
        ]
        cross_scale_context = sum(
            scale_weights[:, index:index + 1] * context
            for index, context in enumerate(scale_contexts)
        )

        routed_skips = []
        spatial_masks = []
        effective_gates = (
            torch.tanh(self.residual_gate) * self.max_residual_strength
        )
        for index, (
            skip,
            local_feature,
            attended,
            channel_head,
            spatial_local,
            spatial_text,
            spatial_context,
        ) in enumerate(zip(
            skips,
            projected,
            attended_text,
            self.channel_heads,
            self.spatial_local,
            self.spatial_text,
            self.spatial_context,
        )):
            joint_context = torch.cat(
                (cross_scale_context, attended),
                dim=1,
            )
            channel_gain = torch.tanh(channel_head(joint_context))
            channel_gain = channel_gain[:, :, None, None]
            spatial_logit = spatial_local(local_feature)
            spatial_logit = spatial_logit + spatial_text(attended)[:, :, None, None]
            spatial_logit = spatial_logit + spatial_context(
                cross_scale_context
            )[:, :, None, None]
            spatial_mask = torch.sigmoid(spatial_logit)
            residual = skip * channel_gain * spatial_mask
            strength = (
                effective_gates[index]
                * scale_weights[:, index:index + 1, None, None]
            )
            routed_skips.append(skip + strength * residual)
            spatial_masks.append(spatial_mask)

        with torch.no_grad():
            scale_mean = scale_weights.mean(dim=0)
            scale_entropy = -(
                scale_weights.clamp_min(1e-8)
                * scale_weights.clamp_min(1e-8).log()
            ).sum(dim=1).mean()
            self._last_stats = {
                "architecture_version": self.architecture_version,
                "scale_weights": [
                    float(value.item()) for value in scale_mean
                ],
                "scale_weight_sum": float(scale_mean.sum().item()),
                "scale_entropy": float(scale_entropy.item()),
                "spatial_mask_means": [
                    float(value.mean().item()) for value in spatial_masks
                ],
                "effective_gates": [
                    float(value.item()) for value in effective_gates
                ],
                "token_attention_entropies": [
                    float((-(
                        weights.clamp_min(1e-8)
                        * weights.clamp_min(1e-8).log()
                    ).sum(dim=1).mean()).item())
                    for weights in token_weights
                ],
            }
        return tuple(routed_skips)


class TextConditionedCrossScaleSkipRouterV2(nn.Module):
    """Spatial cross-scale exchange with independent text-conditioned routes.

    V1 compressed every scale to one vector, forced the scales to compete via
    a shared softmax, and could only reweight each skip multiplicatively.  V2
    keeps cross-scale information spatial: every scale exchanges projected
    features with its immediate neighbours, conditions that consensus with
    masked token attention, and injects a learned residual through an
    independent confidence gate.  Fixed 2x average pooling and nearest-neighbour
    upsampling keep the formal deterministic CUDA protocol intact.
    """

    architecture_version = "tcsr_v2"

    def __init__(
        self,
        skip_channels,
        text_dim=768,
        routing_dim=32,
        max_residual_strength=0.5,
        initial_residual_strength=0.05,
    ):
        super().__init__()
        self.skip_channels = tuple(int(value) for value in skip_channels)
        self.num_scales = len(self.skip_channels)
        self.text_dim = int(text_dim)
        self.routing_dim = int(routing_dim)
        self.max_residual_strength = float(max_residual_strength)
        self.initial_residual_strength = float(initial_residual_strength)
        if self.num_scales < 2:
            raise ValueError("TCSR V2 requires at least two skip scales.")
        if self.routing_dim <= 0:
            raise ValueError("routing_dim must be positive.")
        if self.max_residual_strength <= 0:
            raise ValueError("max_residual_strength must be positive.")
        if not 0 < self.initial_residual_strength < self.max_residual_strength:
            raise ValueError(
                "initial_residual_strength must be between zero and the "
                "maximum residual strength."
            )

        self.visual_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, self.routing_dim, kernel_size=1),
                nn.GroupNorm(1, self.routing_dim),
                nn.SiLU(),
            )
            for channels in self.skip_channels
        ])
        neighbour_counts = [
            1 + int(index > 0) + int(index + 1 < self.num_scales)
            for index in range(self.num_scales)
        ]
        self.consensus_fusions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    count * self.routing_dim,
                    self.routing_dim,
                    kernel_size=1,
                ),
                nn.GroupNorm(1, self.routing_dim),
                nn.SiLU(),
                nn.Conv2d(
                    self.routing_dim,
                    self.routing_dim,
                    kernel_size=3,
                    padding=1,
                    groups=self.routing_dim,
                ),
                nn.GroupNorm(1, self.routing_dim),
                nn.SiLU(),
            )
            for count in neighbour_counts
        ])
        self.visual_norms = nn.ModuleList([
            nn.LayerNorm(self.routing_dim)
            for _ in self.skip_channels
        ])

        self.text_key = nn.Linear(self.text_dim, self.routing_dim)
        self.text_value = nn.Linear(self.text_dim, self.routing_dim)
        self.text_norms = nn.ModuleList([
            nn.LayerNorm(self.routing_dim)
            for _ in self.skip_channels
        ])

        joint_dim = self.routing_dim * 2
        score_dim = self.routing_dim * 3
        score_width = max(16, self.routing_dim)
        self.route_confidence_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(score_dim, score_width),
                nn.SiLU(),
                nn.Linear(score_width, 1),
            )
            for _ in self.skip_channels
        ])
        for head in self.route_confidence_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

        self.film_heads = nn.ModuleList([
            nn.Linear(joint_dim, self.routing_dim * 2)
            for _ in self.skip_channels
        ])
        for head in self.film_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        self.channel_heads = nn.ModuleList([
            nn.Linear(joint_dim, channels)
            for channels in self.skip_channels
        ])
        self.spatial_heads = nn.ModuleList([
            nn.Conv2d(self.routing_dim, 1, kernel_size=3, padding=1)
            for _ in self.skip_channels
        ])
        for head in self.spatial_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        self.message_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.routing_dim, channels, kernel_size=1),
                nn.GroupNorm(1, channels),
            )
            for channels in self.skip_channels
        ])

        initial_fraction = (
            self.initial_residual_strength / self.max_residual_strength
        )
        initial_logit = math.log(initial_fraction / (1.0 - initial_fraction))
        self.residual_strength_logit = nn.Parameter(
            torch.full((self.num_scales,), initial_logit)
        )
        self._last_stats = None

    @staticmethod
    def _validated_text_mask(text, text_mask):
        return TextConditionedCrossScaleSkipRouter._validated_text_mask(
            text,
            text_mask,
        )

    @staticmethod
    def _align_adjacent(feature, target_shape):
        source_shape = tuple(feature.shape[-2:])
        target_shape = tuple(int(value) for value in target_shape)
        if source_shape == target_shape:
            return feature
        if (
            source_shape[0] == target_shape[0] * 2
            and source_shape[1] == target_shape[1] * 2
        ):
            return F.avg_pool2d(feature, kernel_size=2, stride=2)
        if (
            source_shape[0] * 2 == target_shape[0]
            and source_shape[1] * 2 == target_shape[1]
        ):
            return F.interpolate(feature, size=target_shape, mode="nearest")
        raise ValueError(
            "TCSR V2 only exchanges adjacent 2x scales; cannot align {} to {}."
            .format(source_shape, target_shape)
        )

    def _attend_text(self, descriptor, text_keys, text_values, text_mask, norm):
        logits = torch.einsum("bd,bld->bl", descriptor, text_keys)
        logits = logits / math.sqrt(self.routing_dim)
        logits = logits.masked_fill(
            ~text_mask,
            torch.finfo(logits.dtype).min,
        )
        weights = torch.softmax(logits, dim=1)
        attended = torch.einsum("bl,bld->bd", weights, text_values)
        return norm(attended), weights

    def forward(self, skips, text, text_mask=None):
        if len(skips) != self.num_scales:
            raise ValueError(
                "Expected {} skip scales, received {}.".format(
                    self.num_scales,
                    len(skips),
                )
            )
        if text.ndim != 3 or text.shape[-1] != self.text_dim:
            raise ValueError(
                "Expected text [B, L, {}], received {}.".format(
                    self.text_dim,
                    tuple(text.shape),
                )
            )

        batch_size = text.shape[0]
        projected = []
        for index, (skip, channels, projection) in enumerate(zip(
            skips,
            self.skip_channels,
            self.visual_projections,
        )):
            if skip.ndim != 4:
                raise ValueError(
                    "Skip {} must be BCHW, received {}.".format(
                        index,
                        tuple(skip.shape),
                    )
                )
            if skip.shape[0] != batch_size or skip.shape[1] != channels:
                raise ValueError(
                    "Skip {} expected [B, {}, H, W], received {}.".format(
                        index,
                        channels,
                        tuple(skip.shape),
                    )
                )
            projected.append(projection(skip))

        consensus_features = []
        descriptors = []
        for index, (local, fusion, norm) in enumerate(zip(
            projected,
            self.consensus_fusions,
            self.visual_norms,
        )):
            target_shape = local.shape[-2:]
            neighbours = [local]
            if index > 0:
                neighbours.append(self._align_adjacent(
                    projected[index - 1],
                    target_shape,
                ))
            if index + 1 < self.num_scales:
                neighbours.append(self._align_adjacent(
                    projected[index + 1],
                    target_shape,
                ))
            consensus = fusion(torch.cat(neighbours, dim=1))
            consensus_features.append(consensus)
            descriptors.append(norm(consensus.mean(dim=(2, 3))))

        mask = self._validated_text_mask(text, text_mask)
        text_keys = self.text_key(text)
        text_values = self.text_value(text)
        attended_text = []
        token_weights = []
        for descriptor, norm in zip(descriptors, self.text_norms):
            attended, weights = self._attend_text(
                descriptor,
                text_keys,
                text_values,
                mask,
                norm,
            )
            attended_text.append(attended)
            token_weights.append(weights)

        routed_skips = []
        route_confidences = []
        spatial_masks = []
        message_rms = []
        delta_rms_ratios = []
        effective_strengths = (
            torch.sigmoid(self.residual_strength_logit)
            * self.max_residual_strength
        )
        for index, (
            skip,
            consensus,
            descriptor,
            attended,
            confidence_head,
            film_head,
            channel_head,
            spatial_head,
            message_head,
        ) in enumerate(zip(
            skips,
            consensus_features,
            descriptors,
            attended_text,
            self.route_confidence_heads,
            self.film_heads,
            self.channel_heads,
            self.spatial_heads,
            self.message_heads,
        )):
            score_input = torch.cat(
                (descriptor, attended, descriptor * attended),
                dim=1,
            )
            confidence = torch.sigmoid(confidence_head(score_input))
            joint = torch.cat((descriptor, attended), dim=1)
            film_scale, film_shift = film_head(joint).chunk(2, dim=1)
            modulated = consensus * (
                1.0 + 0.5 * torch.tanh(film_scale)[:, :, None, None]
            )
            modulated = modulated + (
                0.5 * torch.tanh(film_shift)[:, :, None, None]
            )
            spatial_mask = torch.sigmoid(spatial_head(modulated))
            channel_delta = torch.tanh(channel_head(joint))[:, :, None, None]
            message = message_head(modulated)
            residual = message + skip * channel_delta
            strength = effective_strengths[index]
            delta = strength * confidence[:, :, None, None] * spatial_mask
            delta = delta * residual
            routed_skips.append(skip + delta)
            route_confidences.append(confidence)
            spatial_masks.append(spatial_mask)
            message_rms.append(message.square().mean().sqrt())
            delta_rms_ratios.append(
                delta.square().mean().sqrt()
                / skip.square().mean().sqrt().clamp_min(1e-8)
            )

        with torch.no_grad():
            self._last_stats = {
                "architecture_version": self.architecture_version,
                "route_confidences": [
                    float(value.mean().item())
                    for value in route_confidences
                ],
                "spatial_mask_means": [
                    float(value.mean().item()) for value in spatial_masks
                ],
                "effective_strengths": [
                    float(value.item()) for value in effective_strengths
                ],
                "message_rms": [
                    float(value.item()) for value in message_rms
                ],
                "delta_rms_ratios": [
                    float(value.item()) for value in delta_rms_ratios
                ],
                "token_attention_entropies": [
                    float((-(
                        weights.clamp_min(1e-8)
                        * weights.clamp_min(1e-8).log()
                    ).sum(dim=1).mean()).item())
                    for weights in token_weights
                ],
            }
        return tuple(routed_skips)


class BoundaryPreservingAsymmetricTextGuidedRouter(nn.Module):
    """Route coarse semantics toward finer skips without rewriting boundaries.

    This V2.1 pilot deliberately leaves the highest-resolution ``x1`` skip
    and the deepest ``x4`` skip unchanged.  It performs only two sequential
    coarse-to-fine routes (``x4 -> x3`` and ``x3 -> x2``), so no fine feature
    is average-pooled into a coarser representation.  Each residual is RMS
    normalized, bounded to a small fraction of the target skip, and controlled
    by a hard-sigmoid abstention gate that can become exactly zero.
    """

    architecture_version = "tcsr_v2_1_boundary_asymmetric"
    route_names = ("x4_to_x3", "x3_to_x2")
    route_pairs = ((3, 2), (2, 1))

    def __init__(
        self,
        skip_channels,
        text_dim=768,
        routing_dim=32,
        max_residual_strength=0.15,
        initial_residual_strength=0.08,
        initial_gate_probability=0.15,
        gate_activation_budget=0.35,
        gate_budget_weight=0.02,
        gate_binary_weight=0.005,
    ):
        super().__init__()
        self.skip_channels = tuple(int(value) for value in skip_channels)
        self.num_scales = len(self.skip_channels)
        self.text_dim = int(text_dim)
        self.routing_dim = int(routing_dim)
        self.max_residual_strength = float(max_residual_strength)
        self.initial_residual_strength = float(initial_residual_strength)
        self.initial_gate_probability = float(initial_gate_probability)
        self.gate_activation_budget = float(gate_activation_budget)
        self.gate_budget_weight = float(gate_budget_weight)
        self.gate_binary_weight = float(gate_binary_weight)
        if self.num_scales != 4:
            raise ValueError("TCSR V2.1 requires exactly four skip scales.")
        if self.routing_dim <= 0:
            raise ValueError("routing_dim must be positive.")
        if not 0 < self.initial_residual_strength < self.max_residual_strength:
            raise ValueError(
                "initial_residual_strength must be between zero and the "
                "maximum residual strength."
            )
        if not 0 < self.initial_gate_probability < 1:
            raise ValueError("initial_gate_probability must be in (0, 1).")
        if not 0 < self.gate_activation_budget < 1:
            raise ValueError("gate_activation_budget must be in (0, 1).")

        self.visual_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, self.routing_dim, kernel_size=1),
                nn.GroupNorm(1, self.routing_dim),
                nn.SiLU(),
            )
            for channels in self.skip_channels
        ])
        self.route_fusions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    2 * self.routing_dim,
                    self.routing_dim,
                    kernel_size=1,
                ),
                nn.GroupNorm(1, self.routing_dim),
                nn.SiLU(),
                nn.Conv2d(
                    self.routing_dim,
                    self.routing_dim,
                    kernel_size=3,
                    padding=1,
                    groups=self.routing_dim,
                ),
                nn.GroupNorm(1, self.routing_dim),
                nn.SiLU(),
            )
            for _ in self.route_pairs
        ])
        self.visual_norms = nn.ModuleList([
            nn.LayerNorm(self.routing_dim)
            for _ in self.route_pairs
        ])
        self.text_key = nn.Linear(self.text_dim, self.routing_dim)
        self.text_value = nn.Linear(self.text_dim, self.routing_dim)
        self.text_norms = nn.ModuleList([
            nn.LayerNorm(self.routing_dim)
            for _ in self.route_pairs
        ])

        joint_dim = self.routing_dim * 2
        score_dim = self.routing_dim * 3
        score_width = max(16, self.routing_dim)
        self.abstention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(score_dim, score_width),
                nn.SiLU(),
                nn.Linear(score_width, 1),
            )
            for _ in self.route_pairs
        ])
        gate_bias = 6.0 * self.initial_gate_probability - 3.0
        for head in self.abstention_heads:
            nn.init.normal_(head[-1].weight, mean=0.0, std=0.01)
            nn.init.constant_(head[-1].bias, gate_bias)

        self.film_heads = nn.ModuleList([
            nn.Linear(joint_dim, self.routing_dim * 2)
            for _ in self.route_pairs
        ])
        self.spatial_heads = nn.ModuleList([
            nn.Conv2d(self.routing_dim, 1, kernel_size=3, padding=1)
            for _ in self.route_pairs
        ])
        self.channel_heads = nn.ModuleList([
            nn.Linear(joint_dim, self.skip_channels[target_index])
            for _, target_index in self.route_pairs
        ])
        self.message_heads = nn.ModuleList([
            nn.Conv2d(
                self.routing_dim,
                self.skip_channels[target_index],
                kernel_size=1,
            )
            for _, target_index in self.route_pairs
        ])
        for heads in (
            self.film_heads,
            self.spatial_heads,
            self.channel_heads,
            self.message_heads,
        ):
            for head in heads:
                nn.init.normal_(head.weight, mean=0.0, std=0.01)
                if head.bias is not None:
                    nn.init.zeros_(head.bias)

        initial_fraction = (
            self.initial_residual_strength / self.max_residual_strength
        )
        initial_logit = math.log(initial_fraction / (1.0 - initial_fraction))
        self.residual_strength_logit = nn.Parameter(
            torch.full((len(self.route_pairs),), initial_logit)
        )
        self._last_stats = None
        self._regularization_terms = None

    @staticmethod
    def _validated_text_mask(text, text_mask):
        return TextConditionedCrossScaleSkipRouter._validated_text_mask(
            text,
            text_mask,
        )

    def _attend_text(self, descriptor, text_keys, text_values, text_mask, norm):
        logits = torch.einsum("bd,bld->bl", descriptor, text_keys)
        logits = logits / math.sqrt(self.routing_dim)
        logits = logits.masked_fill(
            ~text_mask,
            torch.finfo(logits.dtype).min,
        )
        weights = torch.softmax(logits, dim=1)
        attended = torch.einsum("bl,bld->bd", weights, text_values)
        return norm(attended), weights

    @staticmethod
    def _rms_normalize_message(message, target):
        message_rms = message.square().mean(
            dim=(1, 2, 3),
            keepdim=True,
        ).sqrt().clamp_min(1e-6)
        target_rms = target.square().mean(
            dim=(1, 2, 3),
            keepdim=True,
        ).sqrt().clamp_min(1e-6)
        return message * (target_rms / message_rms)

    def regularization_loss(self):
        """Return differentiable gate-budget and binarization penalties."""
        if not self._regularization_terms:
            return self.residual_strength_logit.sum() * 0.0
        return self._regularization_terms["total"]

    def forward(self, skips, text, text_mask=None):
        if len(skips) != self.num_scales:
            raise ValueError(
                "Expected {} skip scales, received {}.".format(
                    self.num_scales,
                    len(skips),
                )
            )
        if text.ndim != 3 or text.shape[-1] != self.text_dim:
            raise ValueError(
                "Expected text [B, L, {}], received {}.".format(
                    self.text_dim,
                    tuple(text.shape),
                )
            )
        batch_size = text.shape[0]
        for index, (skip, channels) in enumerate(zip(
            skips,
            self.skip_channels,
        )):
            if skip.ndim != 4:
                raise ValueError(
                    "Skip {} must be BCHW, received {}.".format(
                        index,
                        tuple(skip.shape),
                    )
                )
            if skip.shape[0] != batch_size or skip.shape[1] != channels:
                raise ValueError(
                    "Skip {} expected [B, {}, H, W], received {}.".format(
                        index,
                        channels,
                        tuple(skip.shape),
                    )
                )

        mask = self._validated_text_mask(text, text_mask)
        text_keys = self.text_key(text)
        text_values = self.text_value(text)
        routed = list(skips)
        route_gates = []
        spatial_masks = []
        delta_rms_ratios = []
        token_weights = []
        effective_strengths = (
            torch.sigmoid(self.residual_strength_logit)
            * self.max_residual_strength
        )

        for route_index, (source_index, target_index) in enumerate(
            self.route_pairs
        ):
            source = routed[source_index]
            target = routed[target_index]
            source_projected = self.visual_projections[source_index](source)
            source_projected = F.interpolate(
                source_projected,
                size=target.shape[-2:],
                mode="nearest",
            )
            target_projected = self.visual_projections[target_index](target)
            context = self.route_fusions[route_index](torch.cat(
                (target_projected, source_projected),
                dim=1,
            ))
            descriptor = self.visual_norms[route_index](
                context.mean(dim=(2, 3))
            )
            attended, weights = self._attend_text(
                descriptor,
                text_keys,
                text_values,
                mask,
                self.text_norms[route_index],
            )
            score_input = torch.cat(
                (descriptor, attended, descriptor * attended),
                dim=1,
            )
            route_gate = F.hardsigmoid(
                self.abstention_heads[route_index](score_input)
            )
            joint = torch.cat((descriptor, attended), dim=1)
            film_scale, film_shift = self.film_heads[route_index](joint).chunk(
                2,
                dim=1,
            )
            modulated = context * (
                1.0 + 0.25 * torch.tanh(film_scale)[:, :, None, None]
            )
            modulated = modulated + (
                0.25 * torch.tanh(film_shift)[:, :, None, None]
            )
            spatial_mask = torch.sigmoid(
                self.spatial_heads[route_index](modulated)
            )
            channel_delta = torch.tanh(
                self.channel_heads[route_index](joint)
            )[:, :, None, None]
            message = self.message_heads[route_index](modulated)
            message = message + target * channel_delta
            normalized_message = self._rms_normalize_message(message, target)
            delta = effective_strengths[route_index]
            delta = delta * route_gate[:, :, None, None] * spatial_mask
            delta = delta * normalized_message
            routed[target_index] = target + delta

            route_gates.append(route_gate)
            spatial_masks.append(spatial_mask)
            delta_rms_ratios.append(
                delta.square().mean().sqrt()
                / target.square().mean().sqrt().clamp_min(1e-8)
            )
            token_weights.append(weights)

        gate_values = torch.cat(route_gates, dim=1)
        gate_mean = gate_values.mean()
        budget_penalty = F.relu(
            gate_mean - self.gate_activation_budget
        ).square()
        binary_penalty = (gate_values * (1.0 - gate_values)).mean()
        total_regularization = (
            self.gate_budget_weight * budget_penalty
            + self.gate_binary_weight * binary_penalty
        )
        self._regularization_terms = {
            "total": total_regularization,
            "budget": budget_penalty,
            "binary": binary_penalty,
        }

        with torch.no_grad():
            self._last_stats = {
                "architecture_version": self.architecture_version,
                "route_names": list(self.route_names),
                "route_gate_means": [
                    float(value.mean().item()) for value in route_gates
                ],
                "route_gate_closed_fractions": [
                    float((value <= 0.01).float().mean().item())
                    for value in route_gates
                ],
                "spatial_mask_means": [
                    float(value.mean().item()) for value in spatial_masks
                ],
                "effective_strengths": [
                    float(value.item()) for value in effective_strengths
                ],
                "delta_rms_ratios": [
                    float(value.item()) for value in delta_rms_ratios
                ],
                "gate_activation_mean": float(gate_mean.item()),
                "gate_budget": self.gate_activation_budget,
                "gate_budget_penalty": float(budget_penalty.item()),
                "gate_binary_penalty": float(binary_penalty.item()),
                "regularization_loss": float(total_regularization.item()),
                "token_attention_entropies": [
                    float((-(
                        weights.clamp_min(1e-8)
                        * weights.clamp_min(1e-8).log()
                    ).sum(dim=1).mean()).item())
                    for weights in token_weights
                ],
                "identity_scales": ["x1", "x4"],
            }
        return tuple(routed)


class SingleHopBoundaryFocusedTextGuidedRouter(nn.Module):
    """Route x3 semantics into x2 while preserving every other encoder skip.

    P1 showed that the x4-to-x3 route absorbed the routing budget and rewrote
    the mid-level representation while the x3-to-x2 route collapsed.  V2.2
    removes that competition: it exposes only the clinically useful x3-to-x2
    hop, keeps x1/x3/x4 bit-identical, uses a smooth sigmoid confidence (so
    the route cannot enter a zero-gradient hard-off state), and focuses the
    spatial residual toward target-feature transitions.  The residual is RMS
    normalized and tightly bounded, so the router can add semantic context
    without replacing x2.
    """

    architecture_version = "tcsr_v2_2_single_hop_boundary_focused"
    route_names = ("x3_to_x2",)
    source_index = 2
    target_index = 1

    def __init__(
        self,
        skip_channels,
        text_dim=768,
        routing_dim=32,
        max_residual_strength=0.08,
        initial_residual_strength=0.04,
        initial_gate_probability=0.25,
        gate_min_probability=0.0,
        gate_max_probability=1.0,
        gate_target_min=0.0,
        gate_target_max=1.0,
        gate_calibration_weight=0.0,
    ):
        super().__init__()
        self.skip_channels = tuple(int(value) for value in skip_channels)
        self.num_scales = len(self.skip_channels)
        self.text_dim = int(text_dim)
        self.routing_dim = int(routing_dim)
        self.max_residual_strength = float(max_residual_strength)
        self.initial_residual_strength = float(initial_residual_strength)
        self.initial_gate_probability = float(initial_gate_probability)
        self.gate_min_probability = float(gate_min_probability)
        self.gate_max_probability = float(gate_max_probability)
        self.gate_target_min = float(gate_target_min)
        self.gate_target_max = float(gate_target_max)
        self.gate_calibration_weight = float(gate_calibration_weight)
        if self.num_scales != 4:
            raise ValueError("TCSR V2.2 requires exactly four skip scales.")
        if self.routing_dim <= 0:
            raise ValueError("routing_dim must be positive.")
        if not 0 < self.initial_residual_strength < self.max_residual_strength:
            raise ValueError(
                "initial_residual_strength must be between zero and the "
                "maximum residual strength."
            )
        if not 0 <= self.gate_min_probability < self.gate_max_probability <= 1:
            raise ValueError("gate probability bounds must satisfy 0 <= min < max <= 1.")
        if not (
            self.gate_min_probability
            < self.initial_gate_probability
            < self.gate_max_probability
        ):
            raise ValueError(
                "initial_gate_probability must be strictly inside the gate bounds."
            )
        if not (
            self.gate_min_probability
            <= self.gate_target_min
            <= self.gate_target_max
            <= self.gate_max_probability
        ):
            raise ValueError("gate target band must lie inside the gate bounds.")
        if self.gate_calibration_weight < 0:
            raise ValueError("gate_calibration_weight must be non-negative.")

        source_channels = self.skip_channels[self.source_index]
        target_channels = self.skip_channels[self.target_index]
        self.source_projection = nn.Sequential(
            nn.Conv2d(source_channels, self.routing_dim, kernel_size=1),
            nn.GroupNorm(1, self.routing_dim),
            nn.SiLU(),
        )
        self.target_projection = nn.Sequential(
            nn.Conv2d(target_channels, self.routing_dim, kernel_size=1),
            nn.GroupNorm(1, self.routing_dim),
            nn.SiLU(),
        )
        self.route_fusion = nn.Sequential(
            nn.Conv2d(2 * self.routing_dim, self.routing_dim, kernel_size=1),
            nn.GroupNorm(1, self.routing_dim),
            nn.SiLU(),
            nn.Conv2d(
                self.routing_dim,
                self.routing_dim,
                kernel_size=3,
                padding=1,
                groups=self.routing_dim,
            ),
            nn.GroupNorm(1, self.routing_dim),
            nn.SiLU(),
        )
        self.visual_norm = nn.LayerNorm(self.routing_dim)
        self.text_key = nn.Linear(self.text_dim, self.routing_dim)
        self.text_value = nn.Linear(self.text_dim, self.routing_dim)
        self.text_norm = nn.LayerNorm(self.routing_dim)

        joint_dim = self.routing_dim * 2
        score_dim = self.routing_dim * 3
        score_width = max(16, self.routing_dim)
        self.route_confidence = nn.Sequential(
            nn.Linear(score_dim, score_width),
            nn.SiLU(),
            nn.Linear(score_width, 1),
        )
        gate_fraction = (
            (self.initial_gate_probability - self.gate_min_probability)
            / (self.gate_max_probability - self.gate_min_probability)
        )
        gate_bias = math.log(gate_fraction / (1.0 - gate_fraction))
        nn.init.normal_(self.route_confidence[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.route_confidence[-1].bias, gate_bias)

        self.film_head = nn.Linear(joint_dim, self.routing_dim * 2)
        self.spatial_head = nn.Conv2d(
            self.routing_dim,
            1,
            kernel_size=3,
            padding=1,
        )
        self.channel_head = nn.Linear(joint_dim, target_channels)
        self.message_head = nn.Conv2d(
            self.routing_dim,
            target_channels,
            kernel_size=1,
        )
        for head in (
            self.film_head,
            self.spatial_head,
            self.channel_head,
            self.message_head,
        ):
            nn.init.normal_(head.weight, mean=0.0, std=0.01)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

        initial_fraction = (
            self.initial_residual_strength / self.max_residual_strength
        )
        initial_logit = math.log(initial_fraction / (1.0 - initial_fraction))
        self.residual_strength_logit = nn.Parameter(
            torch.tensor(initial_logit, dtype=torch.float32)
        )
        self._last_stats = None
        self._regularization_loss = self.residual_strength_logit * 0.0
        self._localization_state = None

    @staticmethod
    def _validated_text_mask(text, text_mask):
        return TextConditionedCrossScaleSkipRouter._validated_text_mask(
            text,
            text_mask,
        )

    @staticmethod
    def _rms_normalize_message(message, target):
        return BoundaryPreservingAsymmetricTextGuidedRouter._rms_normalize_message(
            message,
            target,
        )

    @staticmethod
    def _boundary_focus(target_projected):
        horizontal = F.pad(
            (target_projected[:, :, :, 1:] - target_projected[:, :, :, :-1]).abs(),
            (0, 1, 0, 0),
        )
        vertical = F.pad(
            (target_projected[:, :, 1:, :] - target_projected[:, :, :-1, :]).abs(),
            (0, 0, 0, 1),
        )
        boundary = (horizontal + vertical).mean(dim=1, keepdim=True)
        boundary_rms = boundary.square().mean(
            dim=(2, 3),
            keepdim=True,
        ).sqrt().clamp_min(1e-6)
        normalized = boundary / boundary_rms
        # Keep a 0.5 identity-support floor while preferentially weighting
        # spatial transitions.  This is deterministic and parameter-free.
        return 0.5 + 0.5 * torch.sigmoid(normalized - 1.0)

    def regularization_loss(self):
        return self._regularization_loss

    def _capture_localization_state(self, spatial_mask, delta):
        """Optional differentiable state hook used by supervised routers."""
        self._localization_state = None

    def forward(self, skips, text, text_mask=None):
        if len(skips) != self.num_scales:
            raise ValueError(
                "Expected {} skip scales, received {}.".format(
                    self.num_scales,
                    len(skips),
                )
            )
        if text.ndim != 3 or text.shape[-1] != self.text_dim:
            raise ValueError(
                "Expected text [B, L, {}], received {}.".format(
                    self.text_dim,
                    tuple(text.shape),
                )
            )
        batch_size = text.shape[0]
        for index, (skip, channels) in enumerate(zip(skips, self.skip_channels)):
            if skip.ndim != 4:
                raise ValueError(
                    "Skip {} must be BCHW, received {}.".format(
                        index,
                        tuple(skip.shape),
                    )
                )
            if skip.shape[0] != batch_size or skip.shape[1] != channels:
                raise ValueError(
                    "Skip {} expected [B, {}, H, W], received {}.".format(
                        index,
                        channels,
                        tuple(skip.shape),
                    )
                )

        source = skips[self.source_index]
        target = skips[self.target_index]
        source_projected = self.source_projection(source)
        source_projected = F.interpolate(
            source_projected,
            size=target.shape[-2:],
            mode="nearest",
        )
        target_projected = self.target_projection(target)
        context = self.route_fusion(torch.cat(
            (target_projected, source_projected),
            dim=1,
        ))
        descriptor = self.visual_norm(context.mean(dim=(2, 3)))

        mask = self._validated_text_mask(text, text_mask)
        text_keys = self.text_key(text)
        text_values = self.text_value(text)
        logits = torch.einsum("bd,bld->bl", descriptor, text_keys)
        logits = logits / math.sqrt(self.routing_dim)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        token_weights = torch.softmax(logits, dim=1)
        attended = torch.einsum("bl,bld->bd", token_weights, text_values)
        attended = self.text_norm(attended)

        score_input = torch.cat(
            (descriptor, attended, descriptor * attended),
            dim=1,
        )
        raw_route_gate = torch.sigmoid(self.route_confidence(score_input))
        route_gate = self.gate_min_probability + (
            self.gate_max_probability - self.gate_min_probability
        ) * raw_route_gate
        gate_mean = route_gate.mean()
        gate_calibration_penalty = self.gate_calibration_weight * (
            F.relu(self.gate_target_min - gate_mean).square()
            + F.relu(gate_mean - self.gate_target_max).square()
        )
        self._regularization_loss = gate_calibration_penalty
        joint = torch.cat((descriptor, attended), dim=1)
        film_scale, film_shift = self.film_head(joint).chunk(2, dim=1)
        modulated = context * (
            1.0 + 0.25 * torch.tanh(film_scale)[:, :, None, None]
        )
        modulated = modulated + (
            0.25 * torch.tanh(film_shift)[:, :, None, None]
        )
        boundary_focus = self._boundary_focus(target_projected)
        spatial_mask = torch.sigmoid(self.spatial_head(modulated))
        spatial_mask = spatial_mask * boundary_focus
        channel_delta = torch.tanh(self.channel_head(joint))[:, :, None, None]
        message = self.message_head(modulated) + target * channel_delta
        normalized_message = self._rms_normalize_message(message, target)
        effective_strength = (
            torch.sigmoid(self.residual_strength_logit)
            * self.max_residual_strength
        )
        delta = effective_strength * route_gate[:, :, None, None]
        delta = delta * spatial_mask * normalized_message
        self._capture_localization_state(spatial_mask, delta)

        routed = list(skips)
        routed[self.target_index] = target + delta
        delta_rms_ratio = (
            delta.square().mean().sqrt()
            / target.square().mean().sqrt().clamp_min(1e-8)
        )
        with torch.no_grad():
            self._last_stats = {
                "architecture_version": self.architecture_version,
                "route_names": list(self.route_names),
                "route_gate_means": [float(route_gate.mean().item())],
                "route_gate_closed_fractions": [
                    float((route_gate <= 0.01).float().mean().item())
                ],
                "spatial_mask_means": [float(spatial_mask.mean().item())],
                "boundary_focus_means": [float(boundary_focus.mean().item())],
                "effective_strengths": [float(effective_strength.item())],
                "delta_rms_ratios": [float(delta_rms_ratio.item())],
                "gate_min_probability": self.gate_min_probability,
                "gate_max_probability": self.gate_max_probability,
                "gate_target_min": self.gate_target_min,
                "gate_target_max": self.gate_target_max,
                "gate_calibration_penalty": float(
                    gate_calibration_penalty.item()
                ),
                "regularization_loss": float(
                    gate_calibration_penalty.item()
                ),
                "token_attention_entropies": [
                    float((- (
                        token_weights.clamp_min(1e-8)
                        * token_weights.clamp_min(1e-8).log()
                    ).sum(dim=1).mean()).item())
                ],
                "identity_scales": ["x1", "x3", "x4"],
            }
        return tuple(routed)


class CalibratedSingleHopBoundaryFocusedTextGuidedRouter(
    SingleHopBoundaryFocusedTextGuidedRouter
):
    """V2.3: retain P2's route while preventing an always-on global gate."""

    architecture_version = "tcsr_v2_3_calibrated_single_hop_gate"

    def __init__(
        self,
        skip_channels,
        text_dim=768,
        routing_dim=32,
        max_residual_strength=0.08,
        initial_residual_strength=0.04,
        initial_gate_probability=0.25,
        gate_min_probability=0.05,
        gate_max_probability=0.50,
        gate_target_min=0.15,
        gate_target_max=0.35,
        gate_calibration_weight=0.01,
    ):
        super().__init__(
            skip_channels=skip_channels,
            text_dim=text_dim,
            routing_dim=routing_dim,
            max_residual_strength=max_residual_strength,
            initial_residual_strength=initial_residual_strength,
            initial_gate_probability=initial_gate_probability,
            gate_min_probability=gate_min_probability,
            gate_max_probability=gate_max_probability,
            gate_target_min=gate_target_min,
            gate_target_max=gate_target_max,
            gate_calibration_weight=gate_calibration_weight,
        )


class SparseBoundaryCalibratedTextGuidedRouter(
    CalibratedSingleHopBoundaryFocusedTextGuidedRouter
):
    """V2.4: make P3's spatial support boundary-selective instead of dense.

    P3 kept a 0.5 floor in its boundary map, producing a mean focus near 0.73
    and applying coarse x3 semantics over most of x2.  This variant changes
    only that fixed focus transform: a sharper no-floor sigmoid suppresses
    non-transition regions while preserving the route, gate, message, loss,
    decoder and optimizer protocol.
    """

    architecture_version = "tcsr_v2_4_sparse_boundary_calibrated_gate"

    @staticmethod
    def _boundary_focus(target_projected):
        horizontal = F.pad(
            (target_projected[:, :, :, 1:] - target_projected[:, :, :, :-1]).abs(),
            (0, 1, 0, 0),
        )
        vertical = F.pad(
            (target_projected[:, :, 1:, :] - target_projected[:, :, :-1, :]).abs(),
            (0, 0, 0, 1),
        )
        boundary = (horizontal + vertical).mean(dim=1, keepdim=True)
        boundary_rms = boundary.square().mean(
            dim=(2, 3),
            keepdim=True,
        ).sqrt().clamp_min(1e-6)
        normalized = boundary / boundary_rms
        # The only P4 variable: remove P2/P3's 0.5 support floor and sharpen
        # the transition threshold so flat regions receive near-zero routing.
        return torch.sigmoid(4.0 * (normalized - 1.5))


class SupervisedLocalSparseBoundaryTextGuidedRouter(
    SparseBoundaryCalibratedTextGuidedRouter
):
    """V2.5: align routing support to label boundaries during training.

    Labels are used only by the auxiliary training objective. Inference keeps
    the exact V2.4 forward path and therefore requires no mask or extra input.
    """

    architecture_version = "tcsr_v2_5_supervised_local_sparse_boundary"

    def __init__(
        self,
        skip_channels,
        text_dim=768,
        routing_dim=32,
        max_residual_strength=0.08,
        initial_residual_strength=0.04,
        initial_gate_probability=0.25,
        gate_min_probability=0.05,
        gate_max_probability=0.50,
        gate_target_min=0.15,
        gate_target_max=0.35,
        gate_calibration_weight=0.01,
        localization_weight=0.02,
        residual_leakage_weight=0.5,
    ):
        super().__init__(
            skip_channels=skip_channels,
            text_dim=text_dim,
            routing_dim=routing_dim,
            max_residual_strength=max_residual_strength,
            initial_residual_strength=initial_residual_strength,
            initial_gate_probability=initial_gate_probability,
            gate_min_probability=gate_min_probability,
            gate_max_probability=gate_max_probability,
            gate_target_min=gate_target_min,
            gate_target_max=gate_target_max,
            gate_calibration_weight=gate_calibration_weight,
        )
        if localization_weight < 0.0:
            raise ValueError("localization_weight must be non-negative")
        if residual_leakage_weight < 0.0:
            raise ValueError(
                "residual_leakage_weight must be non-negative"
            )
        self.localization_weight = float(localization_weight)
        self.residual_leakage_weight = float(residual_leakage_weight)
        self._last_localization_components = {}

    def _capture_localization_state(self, spatial_mask, delta):
        if self.training:
            self._localization_state = {
                "spatial_mask": spatial_mask,
                "delta_energy": delta.abs().mean(dim=1, keepdim=True),
            }
        else:
            self._localization_state = None

    @staticmethod
    def _target_boundary(targets, size):
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        if targets.ndim != 4 or targets.shape[1] != 1:
            raise ValueError(
                "Expected binary masks [B, 1, H, W], received {}."
                .format(tuple(targets.shape))
            )
        targets = (targets > 0.5).to(dtype=torch.float32)
        targets = F.interpolate(targets, size=size, mode="nearest")
        dilated = F.max_pool2d(targets, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(
            -targets,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        return (dilated - eroded).clamp(0.0, 1.0)

    def supervised_localization_loss(self, targets, weight_scale=1.0):
        if not 0.0 <= float(weight_scale) <= 1.0:
            raise ValueError("weight_scale must be in [0, 1]")
        if not self._localization_state:
            return self.residual_strength_logit.sum() * 0.0
        spatial_mask = self._localization_state["spatial_mask"]
        delta_energy = self._localization_state["delta_energy"]
        boundary = self._target_boundary(
            targets,
            spatial_mask.shape[-2:],
        ).to(device=spatial_mask.device, dtype=spatial_mask.dtype)
        smooth = 1e-5
        mask_flat = spatial_mask.flatten(1)
        boundary_flat = boundary.flatten(1)
        intersection = (mask_flat * boundary_flat).sum(dim=1)
        mask_dice = (
            2.0 * intersection + smooth
        ) / (
            mask_flat.sum(dim=1) + boundary_flat.sum(dim=1) + smooth
        )
        mask_alignment_loss = 1.0 - mask_dice.mean()

        energy_flat = delta_energy.flatten(1)
        outside_flat = (1.0 - boundary).flatten(1)
        residual_leakage = (
            (energy_flat * outside_flat).sum(dim=1)
            / energy_flat.sum(dim=1).clamp_min(smooth)
        ).mean()
        unweighted = (
            mask_alignment_loss
            + self.residual_leakage_weight * residual_leakage
        )
        total = (
            self.localization_weight * float(weight_scale) * unweighted
        )
        self._last_localization_components = {
            "tcsr_mask_alignment": mask_alignment_loss.detach(),
            "tcsr_residual_leakage": residual_leakage.detach(),
            "tcsr_boundary_fraction": boundary.mean().detach(),
            "tcsr_localization": total.detach(),
            "tcsr_localization_weight_scale": total.new_tensor(
                float(weight_scale)
            ),
        }
        return total

    def localization_components(self):
        return dict(self._last_localization_components)

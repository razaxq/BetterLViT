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

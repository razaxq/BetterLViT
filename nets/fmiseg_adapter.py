# -*- coding: utf-8 -*-
"""A controlled FMISeg-inspired decoder adapter for BetterLViT.

This is not a reproduction of the published dual-ConvNeXt FMISeg network. It
adapts the paper's two defining operations to BetterLViT decoder skips:

1. FFBI-style bidirectional interaction between Haar low/high-frequency tokens.
2. LFFI-style bidirectional interaction between visual and CXR-BERT tokens.

Spatial attention is computed on a fixed 7x7 token grid, keeping the ablation
practical on the local AMD GPU while all output corrections are zero at
initialization. The module therefore starts from the unfiltered
``skip + PLAM`` tensor before learning frequency/text corrections.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .eppa import FixedHaarFrequencySplit


def _bounded_strength_logit(initial, maximum):
    if not 0.0 < initial < maximum:
        raise ValueError("initial strength must be between zero and maximum")
    ratio = initial / maximum
    return math.log(ratio / (1.0 - ratio))


class FMISegDecoderAdapter(nn.Module):
    """FMISeg-inspired frequency/text interaction on one decoder skip."""

    def __init__(
        self,
        channels,
        text_dim=768,
        attention_dim=32,
        attention_heads=4,
        pool_size=7,
        strength_init=0.10,
        strength_max=0.50,
    ):
        super().__init__()
        if attention_dim % attention_heads:
            raise ValueError("attention_dim must be divisible by attention_heads")

        self.channels = int(channels)
        self.attention_dim = int(attention_dim)
        self.pool_size = int(pool_size)
        self.strength_max = float(strength_max)
        self.frequency_split = FixedHaarFrequencySplit(channels)

        self.low_projection = nn.Conv2d(channels, attention_dim, 1, bias=False)
        self.high_projection = nn.Conv2d(channels, attention_dim, 1, bias=False)
        self.text_projection = nn.Linear(text_dim, attention_dim, bias=False)

        self.high_from_low = nn.MultiheadAttention(
            attention_dim,
            attention_heads,
            batch_first=True,
        )
        self.low_from_high = nn.MultiheadAttention(
            attention_dim,
            attention_heads,
            batch_first=True,
        )
        self.text_from_low = nn.MultiheadAttention(
            attention_dim,
            attention_heads,
            batch_first=True,
        )
        self.low_from_text = nn.MultiheadAttention(
            attention_dim,
            attention_heads,
            batch_first=True,
        )
        self.text_from_high = nn.MultiheadAttention(
            attention_dim,
            attention_heads,
            batch_first=True,
        )
        self.high_from_text = nn.MultiheadAttention(
            attention_dim,
            attention_heads,
            batch_first=True,
        )

        self.low_norm = nn.LayerNorm(attention_dim)
        self.high_norm = nn.LayerNorm(attention_dim)
        self.text_low_norm = nn.LayerNorm(attention_dim)
        self.text_high_norm = nn.LayerNorm(attention_dim)

        self.low_filter = nn.Conv2d(attention_dim, channels, 1)
        self.high_filter = nn.Conv2d(attention_dim, channels, 1)
        nn.init.zeros_(self.low_filter.weight)
        nn.init.zeros_(self.low_filter.bias)
        nn.init.zeros_(self.high_filter.weight)
        nn.init.zeros_(self.high_filter.bias)

        self.strength_logit = nn.Parameter(
            torch.tensor(
                _bounded_strength_logit(strength_init, strength_max)
            ).view(1, 1, 1, 1)
        )
        self._last_stats = {}

    def _visual_tokens(self, features, projection):
        projected = projection(features)
        pooled = F.adaptive_avg_pool2d(
            projected,
            (self.pool_size, self.pool_size),
        )
        return pooled.flatten(2).transpose(1, 2)

    def _token_map(self, tokens, output_size):
        batch = tokens.shape[0]
        token_map = tokens.transpose(1, 2).reshape(
            batch,
            self.attention_dim,
            self.pool_size,
            self.pool_size,
        )
        return F.interpolate(
            token_map,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, skip, plam, text, text_mask=None):
        if plam is None:
            plam = torch.zeros_like(skip)
        visual = skip + plam
        low, high = self.frequency_split(visual)

        low_tokens = self._visual_tokens(low, self.low_projection)
        high_tokens = self._visual_tokens(high, self.high_projection)

        high_context, _ = self.high_from_low(
            high_tokens,
            low_tokens,
            low_tokens,
            need_weights=False,
        )
        low_context, _ = self.low_from_high(
            low_tokens,
            high_tokens,
            high_tokens,
            need_weights=False,
        )
        high_tokens = self.high_norm(high_tokens + high_context)
        low_tokens = self.low_norm(low_tokens + low_context)

        text_tokens = self.text_projection(text)
        key_padding_mask = None
        if text_mask is not None:
            key_padding_mask = ~text_mask.to(dtype=torch.bool)

        text_low, _ = self.text_from_low(
            text_tokens,
            low_tokens,
            low_tokens,
            need_weights=False,
        )
        text_low = self.text_low_norm(text_tokens + text_low)
        low_text, _ = self.low_from_text(
            low_tokens,
            text_low,
            text_low,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        low_tokens = self.low_norm(low_tokens + low_text)

        text_high, _ = self.text_from_high(
            text_tokens,
            high_tokens,
            high_tokens,
            need_weights=False,
        )
        text_high = self.text_high_norm(text_tokens + text_high)
        high_text, _ = self.high_from_text(
            high_tokens,
            text_high,
            text_high,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        high_tokens = self.high_norm(high_tokens + high_text)

        output_size = visual.shape[-2:]
        low_map = self._token_map(low_tokens, output_size)
        high_map = self._token_map(high_tokens, output_size)
        low_gain = 1.0 + 0.5 * torch.tanh(self.low_filter(low_map))
        high_gain = 1.0 + 0.5 * torch.tanh(self.high_filter(high_map))
        frequency_refined = low * low_gain + high * high_gain
        correction = frequency_refined - visual
        strength = self.strength_max * torch.sigmoid(self.strength_logit)
        output = visual + strength * correction

        if not self.training:
            with torch.no_grad():
                self._last_stats = {
                    "architecture_version": "fmiseg_adapter_v1",
                    "strength": float(strength.mean().item()),
                    "haar_reconstruction_error": float(
                        (visual - low - high).abs().max().item()
                    ),
                    "low_gain_mean": float(low_gain.mean().item()),
                    "low_gain_std": float(low_gain.std().item()),
                    "high_gain_mean": float(high_gain.mean().item()),
                    "high_gain_std": float(high_gain.std().item()),
                    "correction_std": float(correction.std().item()),
                }
        return output

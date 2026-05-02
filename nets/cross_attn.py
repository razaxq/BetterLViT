# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    """Multi-head cross-attention: image patches as Q, text tokens as K/V.

    PAD positions in `ctx_mask` are masked out before softmax so they
    contribute zero weight. The output projection is zero-initialised, so the
    module is an exact no-op at training start (output ≡ 0) but every weight
    receives meaningful gradient through the residual connection — this avoids
    the cold-start trap of a tanh gate (whose gradient depends on the random
    cross-attn output and can stay near zero indefinitely).
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, \
            'embed_dim {} not divisible by num_heads {}'.format(dim, num_heads)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Zero-init the output projection: cross-attn output starts at exactly 0,
        # so the residual `x = x + cross_attn(...)` is identity at init, and the
        # weights grow organically as the network finds them useful.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, ctx, ctx_mask=None):
        """
        Args:
            x:        [B, N, C]   query (e.g. patch tokens, N=196)
            ctx:      [B, M, C]   key/value (text tokens, M=text_seq_len)
            ctx_mask: [B, M] long/bool   1 (or True) = valid, 0 (or False) = PAD
        Returns:
            [B, N, C]   cross-attended output (caller is expected to add as residual)
        """
        B, N, C = x.shape
        M = ctx.shape[1]

        Q = self.q(self.norm_q(x))                           # [B, N, C]
        KV = self.kv(self.norm_kv(ctx))                      # [B, M, 2C]
        K, V = KV.chunk(2, dim=-1)                           # each [B, M, C]

        Q = Q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale        # [B, H, N, M]

        if ctx_mask is not None:
            # broadcast [B, M] -> [B, 1, 1, M] over heads and queries
            mask = ctx_mask[:, None, None, :].to(dtype=torch.bool)
            attn = attn.masked_fill(~mask, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ V                                       # [B, H, N, hd]
        out = out.transpose(1, 2).reshape(B, N, C)           # [B, N, C]
        out = self.proj(out)                                 # zero at init
        out = self.proj_drop(out)
        return out

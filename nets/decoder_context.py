"""Matched visual/text context controls at the intermediate decoder output."""
import torch
from torch import nn


class DecoderContextAdapter(nn.Module):
    """128-channel queries, 32 context tokens, four heads of width eight.

    Only the output projection starts at zero. No trainable residual gate or
    dropout is introduced. This is a control, not a novelty claim.
    """
    def __init__(self, mode):
        super().__init__()
        if mode not in ('visual', 'text'):
            raise ValueError('Expected visual or text context')
        self.mode = mode
        self.query_norm = nn.LayerNorm(128)
        self.context_norm = nn.LayerNorm(128)
        self.query = nn.Linear(128, 32, bias=False)
        self.key = nn.Linear(128, 32, bias=False)
        self.value = nn.Linear(128, 32, bias=False)
        self.output = nn.Linear(32, 128, bias=False)
        nn.init.zeros_(self.output.weight)
        self.scale = 0.1
        self.capture_stats = False
        self.last_stats = {}

    def forward(self, feature, text=None, text_mask=None):
        b, c, h, w = feature.shape
        if c != 128 or h % 4 or w % 8:
            raise ValueError('Expected 128 channels and dimensions divisible by 4 x 8')
        queries = feature.flatten(2).transpose(1, 2)
        if self.mode == 'visual':
            # Equal-sized bins: equivalent to adaptive pooling at 56 x 56,
            # while keeping backward deterministic on CUDA. Never use text mask.
            context = feature.reshape(b, c, 4, h // 4, 8, w // 8).mean((3, 5))
            context = context.flatten(2).transpose(1, 2)
            valid = torch.ones((b, 32), dtype=torch.bool, device=feature.device)
            active = valid.any(1)
        else:
            if text is None or text_mask is None or text.shape != (b, 32, 128) or text_mask.shape != (b, 32):
                raise ValueError('Text control requires B x 32 x 128 states and B x 32 mask')
            valid = text_mask.bool()
            # The fixed CXR-BERT tokenizer contributes CLS and SEP. Keep them
            # as keys for real reports, but an empty CLS/SEP-only report adds 0.
            active = valid.sum(1) > 2
            valid = valid & active[:, None]
            context = text.masked_fill(~valid[:, :, None], 0)
        safe_valid = valid | ~active[:, None]
        q = self.query(self.query_norm(queries)).reshape(b, h * w, 4, 8).transpose(1, 2)
        normalized = self.context_norm(context)
        k = self.key(normalized).reshape(b, 32, 4, 8).transpose(1, 2)
        v = self.value(normalized).reshape(b, 32, 4, 8).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * (8 ** -0.5)
        scores = scores.masked_fill(~safe_valid[:, None, None, :], float('-inf'))
        weights = scores.softmax(-1)
        mixed = (weights @ v).transpose(1, 2).reshape(b, h * w, 32)
        residual = self.output(mixed).transpose(1, 2).reshape(b, c, h, w)
        delta = self.scale * residual * active[:, None, None, None]
        if self.capture_stats:
            with torch.no_grad():
                self.last_stats = dict(
                    residual_rms=float(delta.square().mean().sqrt()),
                    feature_rms=float(feature.square().mean().sqrt()),
                    attention_entropy=float(-(weights * weights.clamp_min(1e-20).log()).sum(-1).mean()),
                    active_reports=int(active.sum()))
        return feature + delta


def adapter_metadata(mode):
    if mode == 'none':
        return None
    if mode in ('regroup_visual', 'regroup_text'):
        return dict(version='decoder_visual_value_regroup_v1', mode=mode,
                    stage='after_up3_before_up2', channels=128, width=32, heads=4,
                    maximum_context_tokens=32, parameters=16896, residual_scale=0.1,
                    output_zero_init=True, extra_loss=False, value_source='image_only',
                    centered_visual_residual=True, shared_forward_reverse_scores=True)
    return dict(version='decoder_context_v1', mode=mode, stage='after_up3_before_up2',
                channels=128, width=32, heads=4, maximum_context_tokens=32,
                parameters=16896, residual_scale=0.1, output_zero_init=True,
                extra_loss=False)


class VisualValueRegroupAdapter(DecoderContextAdapter):
    """Context chooses visual groups; no context vector is used as a value.

    A normalizes scores over anchors; B normalizes the same scores over image
    locations. Prototypes = B^T V_image and messages = A Prototypes - V_image.
    Both variants have exactly the same parameters as the original control.
    This is an exploratory mechanism, not a claim of a new attention primitive.
    """
    def __init__(self, mode):
        if mode not in ('regroup_visual', 'regroup_text'):
            raise ValueError('Unknown regroup mode')
        super().__init__('text' if mode == 'regroup_text' else 'visual')
        self.regroup_mode = mode

    def forward(self, feature, text=None, text_mask=None):
        b, c, h, w = feature.shape
        if c != 128 or h % 4 or w % 8:
            raise ValueError('Expected 128 channels, dimensions divisible by 4 x 8')
        pixels = feature.flatten(2).transpose(1, 2)
        if self.mode == 'visual':
            context = feature.reshape(b,c,4,h//4,8,w//8).mean((3,5)).flatten(2).transpose(1,2)
            valid = torch.ones((b,32),dtype=torch.bool,device=feature.device)
            active = valid.any(1)
        else:
            if text is None or text_mask is None or text.shape != (b,32,128) or text_mask.shape != (b,32):
                raise ValueError('Expected B x 32 x 128 text and B x 32 mask')
            valid = text_mask.bool()
            active = valid.sum(1) > 2
            valid = valid & active[:,None]
            context = text.masked_fill(~valid[:,:,None],0)
        normalized_pixels = self.query_norm(pixels)
        q = self.query(normalized_pixels).reshape(b,h*w,4,8).transpose(1,2)
        k = self.key(self.context_norm(context)).reshape(b,32,4,8).transpose(1,2)
        values = self.value(normalized_pixels).reshape(b,h*w,4,8).transpose(1,2)
        scores = (q @ k.transpose(-2,-1)) * (8 ** -0.5)
        safe_valid = valid | ~active[:,None]
        to_anchor = scores.masked_fill(~safe_valid[:,None,None,:],float('-inf')).softmax(-1)
        to_anchor = to_anchor * valid[:,None,None,:]
        # Normalize BEFORE masking to avoid all-inf spatial columns at PADs.
        to_pixel = scores.softmax(-2) * valid[:,None,None,:]
        prototypes = to_pixel.transpose(-2,-1) @ values
        regrouped = to_anchor @ prototypes
        message = (regrouped-values) * active[:,None,None,None]
        message = message.transpose(1,2).reshape(b,h*w,32)
        delta = self.scale * self.output(message).transpose(1,2).reshape(b,c,h,w)
        if self.capture_stats:
            with torch.no_grad():
                self.last_stats = dict(
                    residual_rms=float(delta.square().mean().sqrt()),
                    feature_rms=float(feature.square().mean().sqrt()),
                    attention_entropy=float(-(to_anchor*to_anchor.clamp_min(1e-20).log()).sum(-1).mean()),
                    reverse_entropy=float(-(to_pixel*to_pixel.clamp_min(1e-20).log()).sum(-2).mean()),
                    active_reports=int(active.sum()),
                    prototype_rms=float(prototypes.square().mean().sqrt()),
                    centered_message_rms=float(message.square().mean().sqrt()))
        return feature + delta

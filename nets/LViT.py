# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from .Vit import VisionTransformer, Reconstruct
from .eppa import EPPA


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UpblockAttention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv,
                 activation='ReLU', text_dim=None, min_bottleneck_channels=8,
                 use_decoder_guide=True, use_dilated_edge=True,
                 use_text_pixel_film=True,
                 normalize_channel_descriptors=True,
                 use_plam_guide=True,
                 channel_strength_max=0.5,
                 pixel_strength_max=0.35, edge_strength_max=0.3,
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
                 token_temperature_init=5.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        # DG-EPPA uses the upsampled decoder feature as a top-down semantic
        # guide for frequency-routed skip refinement.
        self.eppa = EPPA(
            in_channels // 2,
            text_dim=text_dim,
            reduction=8,
            min_bottleneck_channels=min_bottleneck_channels,
            use_decoder_guide=use_decoder_guide,
            use_dilated_edge=use_dilated_edge,
            use_text_pixel_film=use_text_pixel_film,
            use_plam_guide=use_plam_guide,
            normalize_channel_descriptors=(
                normalize_channel_descriptors
            ),
            channel_strength_max=channel_strength_max,
            pixel_strength_max=pixel_strength_max,
            edge_strength_max=edge_strength_max,
            plam_strength_max=plam_strength_max,
            plam_strength_init=plam_strength_init,
            plam_strength_floor=plam_strength_floor,
            detail_strength_floor=detail_strength_floor,
            use_adaptive_frequency=use_adaptive_frequency,
            frequency_groups=frequency_groups,
            frequency_context_channels=frequency_context_channels,
            alpf_strength_max=alpf_strength_max,
            alpf_strength_init=alpf_strength_init,
            ahpf_strength_max=ahpf_strength_max,
            ahpf_strength_init=ahpf_strength_init,
            ahpf_strength_floor=ahpf_strength_floor,
            use_semantic_flow_alignment=use_semantic_flow_alignment,
            flow_groups=flow_groups,
            flow_max_offset=flow_max_offset,
            flow_strength_max=flow_strength_max,
            flow_strength_init=flow_strength_init,
            use_token_routing=use_token_routing,
            token_attention_dim=token_attention_dim,
            token_attention_heads=token_attention_heads,
            token_strength_max=token_strength_max,
            token_strength_init=token_strength_init,
            token_temperature_init=token_temperature_init,
        )
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(
        self,
        x,
        skip_x,
        plam_x=None,
        text=None,
        text_mask=None,
    ):
        up = self.up(x)
        skip_x_att, up = self.eppa(
            skip_x,
            plam=plam_x,
            decoder=up,
            text=text,
            text_mask=text_mask,
            return_decoder=True,
        )
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


class LViT(nn.Module):
    def __init__(self, config, n_channels=3, n_classes=1, img_size=224, vis=False, text_seq_len=10):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.downVit = VisionTransformer(config, vis, img_size=224, channel_num=64, patch_size=16, embed_dim=64, text_seq_len=text_seq_len)
        self.downVit1 = VisionTransformer(config, vis, img_size=112, channel_num=128, patch_size=8, embed_dim=128, text_seq_len=text_seq_len)
        self.downVit2 = VisionTransformer(config, vis, img_size=56, channel_num=256, patch_size=4, embed_dim=256, text_seq_len=text_seq_len)
        self.downVit3 = VisionTransformer(config, vis, img_size=28, channel_num=512, patch_size=2, embed_dim=512, text_seq_len=text_seq_len)
        self.upVit = VisionTransformer(config, vis, img_size=224, channel_num=64, patch_size=16, embed_dim=64, text_seq_len=text_seq_len)
        self.upVit1 = VisionTransformer(config, vis, img_size=112, channel_num=128, patch_size=8, embed_dim=128, text_seq_len=text_seq_len)
        self.upVit2 = VisionTransformer(config, vis, img_size=56, channel_num=256, patch_size=4, embed_dim=256, text_seq_len=text_seq_len)
        self.upVit3 = VisionTransformer(config, vis, img_size=28, channel_num=512, patch_size=2, embed_dim=512, text_seq_len=text_seq_len)
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)
        TEXT_DIM = 768
        # Per-stage EPPA bottleneck floor, in (up4, up3, up2, up1) order.
        # Default 8 reproduces the legacy EPPA c_red formula; 32 widens the
        # bottleneck for shallow stages where channel discrimination matters most.
        # WARNING: changing any element triggers Linear shape mismatches against
        # checkpoints trained with a different value. EPPA is train-from-scratch
        # only (CLAUDE.md) -- do not set Config.resume_path to an EPPA checkpoint
        # trained with a different EPPA_MIN_BOTTLENECK_CHANNELS.
        EPPA_MIN_BOTTLENECK_CHANNELS = (32, 32, 32, 32)
        EPPA_USE_DECODER_GUIDE = getattr(
            config,
            'eppa_use_decoder_guide',
            True,
        )
        EPPA_USE_DILATED_EDGE = getattr(
            config,
            'eppa_use_dilated_edge',
            True,
        )
        EPPA_USE_TEXT_PIXEL_FILM = getattr(
            config,
            'eppa_use_text_pixel_film',
            True,
        )
        EPPA_USE_PLAM_GUIDE = getattr(
            config,
            'eppa_use_plam_guide',
            True,
        )
        EPPA_NORMALIZE_CHANNEL_DESCRIPTORS = getattr(
            config,
            'eppa_normalize_channel_descriptors',
            True,
        )
        EPPA_CHANNEL_STRENGTH_MAX = getattr(
            config,
            'eppa_channel_strength_max',
            0.5,
        )
        EPPA_PIXEL_STRENGTH_MAX = getattr(
            config,
            'eppa_pixel_strength_max',
            0.35,
        )
        EPPA_EDGE_STRENGTH_MAX = getattr(
            config,
            'eppa_edge_strength_max',
            0.3,
        )
        EPPA_PLAM_STRENGTH_MAX = getattr(
            config,
            'eppa_plam_strength_max',
            1.25,
        )
        EPPA_PLAM_STRENGTH_INIT = getattr(
            config,
            'eppa_plam_strength_init',
            1.0,
        )
        EPPA_PLAM_STRENGTH_FLOOR = getattr(
            config,
            'eppa_plam_strength_floor',
            0.25,
        )
        EPPA_DETAIL_STRENGTH_FLOOR = getattr(
            config,
            'eppa_detail_strength_floor',
            0.02,
        )
        EPPA_ADAPTIVE_FREQUENCY_STAGES = tuple(getattr(
            config,
            'eppa_adaptive_frequency_stages',
            (),
        ))
        EPPA_FREQUENCY_GROUPS = getattr(
            config,
            'eppa_frequency_groups',
            8,
        )
        EPPA_FREQUENCY_CONTEXT_CHANNELS = getattr(
            config,
            'eppa_frequency_context_channels',
            32,
        )
        EPPA_ALPF_STRENGTH_MAX = getattr(
            config,
            'eppa_alpf_strength_max',
            0.50,
        )
        EPPA_ALPF_STRENGTH_INIT = getattr(
            config,
            'eppa_alpf_strength_init',
            0.20,
        )
        EPPA_AHPF_STRENGTH_MAX = getattr(
            config,
            'eppa_ahpf_strength_max',
            0.30,
        )
        EPPA_AHPF_STRENGTH_INIT = getattr(
            config,
            'eppa_ahpf_strength_init',
            0.08,
        )
        EPPA_AHPF_STRENGTH_FLOOR = getattr(
            config,
            'eppa_ahpf_strength_floor',
            0.02,
        )
        EPPA_SEMANTIC_FLOW_STAGES = tuple(getattr(
            config,
            'eppa_semantic_flow_stages',
            (),
        ))
        EPPA_FLOW_GROUPS = getattr(
            config,
            'eppa_flow_groups',
            4,
        )
        EPPA_FLOW_MAX_OFFSET = getattr(
            config,
            'eppa_flow_max_offset',
            1.5,
        )
        EPPA_FLOW_STRENGTH_MAX = getattr(
            config,
            'eppa_flow_strength_max',
            1.0,
        )
        EPPA_FLOW_STRENGTH_INIT = getattr(
            config,
            'eppa_flow_strength_init',
            0.25,
        )
        EPPA_TOKEN_ROUTING_STAGES = tuple(getattr(
            config,
            'eppa_token_routing_stages',
            (),
        ))
        EPPA_TOKEN_ATTENTION_DIM = getattr(
            config,
            'eppa_token_attention_dim',
            32,
        )
        EPPA_TOKEN_ATTENTION_HEADS = getattr(
            config,
            'eppa_token_attention_heads',
            4,
        )
        EPPA_TOKEN_STRENGTH_MAX = getattr(
            config,
            'eppa_token_strength_max',
            0.50,
        )
        EPPA_TOKEN_STRENGTH_INIT = getattr(
            config,
            'eppa_token_strength_init',
            0.10,
        )
        EPPA_TOKEN_TEMPERATURE_INIT = getattr(
            config,
            'eppa_token_temperature_init',
            5.0,
        )
        eppa_common = {
            'text_dim': TEXT_DIM,
            'use_decoder_guide': EPPA_USE_DECODER_GUIDE,
            'use_dilated_edge': EPPA_USE_DILATED_EDGE,
            'use_text_pixel_film': EPPA_USE_TEXT_PIXEL_FILM,
            'use_plam_guide': EPPA_USE_PLAM_GUIDE,
            'normalize_channel_descriptors': (
                EPPA_NORMALIZE_CHANNEL_DESCRIPTORS
            ),
            'channel_strength_max': EPPA_CHANNEL_STRENGTH_MAX,
            'pixel_strength_max': EPPA_PIXEL_STRENGTH_MAX,
            'edge_strength_max': EPPA_EDGE_STRENGTH_MAX,
            'plam_strength_max': EPPA_PLAM_STRENGTH_MAX,
            'plam_strength_init': EPPA_PLAM_STRENGTH_INIT,
            'plam_strength_floor': EPPA_PLAM_STRENGTH_FLOOR,
            'detail_strength_floor': EPPA_DETAIL_STRENGTH_FLOOR,
            'frequency_groups': EPPA_FREQUENCY_GROUPS,
            'frequency_context_channels': (
                EPPA_FREQUENCY_CONTEXT_CHANNELS
            ),
            'alpf_strength_max': EPPA_ALPF_STRENGTH_MAX,
            'alpf_strength_init': EPPA_ALPF_STRENGTH_INIT,
            'ahpf_strength_max': EPPA_AHPF_STRENGTH_MAX,
            'ahpf_strength_init': EPPA_AHPF_STRENGTH_INIT,
            'ahpf_strength_floor': EPPA_AHPF_STRENGTH_FLOOR,
            'flow_groups': EPPA_FLOW_GROUPS,
            'flow_max_offset': EPPA_FLOW_MAX_OFFSET,
            'flow_strength_max': EPPA_FLOW_STRENGTH_MAX,
            'flow_strength_init': EPPA_FLOW_STRENGTH_INIT,
            'token_attention_dim': EPPA_TOKEN_ATTENTION_DIM,
            'token_attention_heads': EPPA_TOKEN_ATTENTION_HEADS,
            'token_strength_max': EPPA_TOKEN_STRENGTH_MAX,
            'token_strength_init': EPPA_TOKEN_STRENGTH_INIT,
            'token_temperature_init': EPPA_TOKEN_TEMPERATURE_INIT,
        }
        self.up4 = UpblockAttention(
            in_channels * 16,
            in_channels * 4,
            nb_Conv=2,
            min_bottleneck_channels=(
                EPPA_MIN_BOTTLENECK_CHANNELS[0]
            ),
            use_adaptive_frequency=(
                'up4' in EPPA_ADAPTIVE_FREQUENCY_STAGES
            ),
            use_semantic_flow_alignment=(
                'up4' in EPPA_SEMANTIC_FLOW_STAGES
            ),
            use_token_routing=(
                'up4' in EPPA_TOKEN_ROUTING_STAGES
            ),
            **eppa_common,
        )
        self.up3 = UpblockAttention(
            in_channels * 8,
            in_channels * 2,
            nb_Conv=2,
            min_bottleneck_channels=(
                EPPA_MIN_BOTTLENECK_CHANNELS[1]
            ),
            use_adaptive_frequency=(
                'up3' in EPPA_ADAPTIVE_FREQUENCY_STAGES
            ),
            use_semantic_flow_alignment=(
                'up3' in EPPA_SEMANTIC_FLOW_STAGES
            ),
            use_token_routing=(
                'up3' in EPPA_TOKEN_ROUTING_STAGES
            ),
            **eppa_common,
        )
        self.up2 = UpblockAttention(
            in_channels * 4,
            in_channels,
            nb_Conv=2,
            min_bottleneck_channels=(
                EPPA_MIN_BOTTLENECK_CHANNELS[2]
            ),
            use_adaptive_frequency=(
                'up2' in EPPA_ADAPTIVE_FREQUENCY_STAGES
            ),
            use_semantic_flow_alignment=(
                'up2' in EPPA_SEMANTIC_FLOW_STAGES
            ),
            use_token_routing=(
                'up2' in EPPA_TOKEN_ROUTING_STAGES
            ),
            **eppa_common,
        )
        self.up1 = UpblockAttention(
            in_channels * 2,
            in_channels,
            nb_Conv=2,
            min_bottleneck_channels=(
                EPPA_MIN_BOTTLENECK_CHANNELS[3]
            ),
            use_adaptive_frequency=(
                'up1' in EPPA_ADAPTIVE_FREQUENCY_STAGES
            ),
            use_semantic_flow_alignment=(
                'up1' in EPPA_SEMANTIC_FLOW_STAGES
            ),
            use_token_routing=(
                'up1' in EPPA_TOKEN_ROUTING_STAGES
            ),
            **eppa_common,
        )
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.last_activation = nn.Sigmoid()  # if using BCELoss
        self.multi_activation = nn.Softmax()
        self.reconstruct1 = Reconstruct(in_channels=64, out_channels=64, kernel_size=1, scale_factor=(16, 16))
        self.reconstruct2 = Reconstruct(in_channels=128, out_channels=128, kernel_size=1, scale_factor=(8, 8))
        self.reconstruct3 = Reconstruct(in_channels=256, out_channels=256, kernel_size=1, scale_factor=(4, 4))
        self.reconstruct4 = Reconstruct(in_channels=512, out_channels=512, kernel_size=1, scale_factor=(2, 2))
        self.text_module4 = nn.Conv1d(in_channels=768, out_channels=512, kernel_size=3, padding=1)
        self.text_module3 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.text_module2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.text_module1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)

    def forward(self, x, text, text_mask=None):
        x = x.float()  # x [4,3,224,224]
        x1 = self.inc(x)  # x1 [4, 64, 224, 224]
        text4 = self.text_module4(text.transpose(1, 2)).transpose(1, 2) 
        text3 = self.text_module3(text4.transpose(1, 2)).transpose(1, 2)
        text2 = self.text_module2(text3.transpose(1, 2)).transpose(1, 2)
        text1 = self.text_module1(text2.transpose(1, 2)).transpose(1, 2)
        y1 = self.downVit(x1, x1, text1)
        x2 = self.down1(x1)
        y2 = self.downVit1(x2, y1, text2)
        x3 = self.down2(x2)
        y3 = self.downVit2(x3, y2, text3)
        x4 = self.down3(x3)
        y4 = self.downVit3(x4, y3, text4)
        x5 = self.down4(x4)
        y4 = self.upVit3(y4, y4, text4, True)
        y3 = self.upVit2(y3, y4, text3, True)
        y2 = self.upVit1(y2, y3, text2, True)
        y1 = self.upVit(y1, y2, text1, True)
        # FAM-EPPA deliberately keeps the local CNN skip and PLAM reconstruction
        # separate until FAM-EPPA performs frequency-aware residual fusion.
        # Earlier experiments added these tensors here, irreversibly mixing
        # local detail and language-conditioned semantics before EPPA.
        plam1 = self.reconstruct1(y1)
        plam2 = self.reconstruct2(y2)
        plam3 = self.reconstruct3(y3)
        plam4 = self.reconstruct4(y4)
        x = self.up4(
            x5, x4, plam4, text=text, text_mask=text_mask
        )
        x = self.up3(
            x, x3, plam3, text=text, text_mask=text_mask
        )
        x = self.up2(
            x, x2, plam2, text=text, text_mask=text_mask
        )
        x = self.up1(
            x, x1, plam1, text=text, text_mask=text_mask
        )
        if self.n_classes == 1:
            logits = self.last_activation(self.outc(x))
        else:
            logits = self.outc(x)  # if not using BCEWithLogitsLoss or class>1
        return logits

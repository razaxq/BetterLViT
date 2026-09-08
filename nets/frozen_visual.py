"""Revision-pinned frozen visual features and an identity-initialized adapter."""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
from transformers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel

ENCODERS = {
    'cxformer': ('m42-health/CXformer-small', '21777c302a43197b704c6c92591d9897efaaac3b'),
    'dinov2': ('facebook/dinov2-with-registers-small', '0d9846e56b43a21fa46d7f3f5070f0506a5795a9'),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


class FrozenVisualEncoder(nn.Module):
    def __init__(self, model_root, kind='cxformer', random_init=False, seed=7001):
        super().__init__()
        if kind not in ENCODERS:
            raise ValueError('Unknown visual encoder')
        root = Path(model_root)
        # Both weight sets use exactly the same effective model configuration.
        reference = json.loads((root / 'dinov2' / 'config.json').read_text())
        source = json.loads((root / kind / 'config.json').read_text())
        for key in ('hidden_size', 'num_hidden_layers', 'num_attention_heads',
                    'num_register_tokens', 'patch_size', 'image_size', 'mlp_ratio'):
            if reference[key] != source[key]:
                raise ValueError('Encoder structural mismatch: ' + key)
        cfg = Dinov2WithRegistersConfig.from_dict(reference)
        cfg._attn_implementation = 'eager'
        devices = list(range(torch.cuda.device_count()))
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            self.backbone = Dinov2WithRegistersModel(cfg)
        weight = root / kind / 'model.safetensors'
        manifest = json.loads((root / 'manifest.json').read_text())
        record = manifest[kind]
        if record['model_id'] != ENCODERS[kind][0] or record['revision'] != ENCODERS[kind][1]:
            raise ValueError('Unregistered encoder revision')
        if sha256(weight) != record['files']['model.safetensors']:
            raise ValueError('External weight checksum mismatch')
        if not random_init:
            self.backbone.load_state_dict(load_file(str(weight)), strict=True)
        self.backbone.requires_grad_(False).eval()
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
        self.provenance = {'kind': kind, 'random_init': bool(random_init),
            'random_seed': seed, 'model_id': record['model_id'], 'revision': record['revision'],
            'weight_sha256': record['files']['model.safetensors'] if not random_init else None,
            'reference_config_sha256': sha256(root / 'dinov2' / 'config.json'),
            'attention': 'eager', 'input': '224 BGR [0,1] -> RGB -> ImageNet normalize',
            'histogram_equalization': False, 'patch_size': 14, 'nonpatch_tokens': 5}

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, image):
        if image.ndim != 4 or tuple(image.shape[1:]) != (3, 224, 224):
            raise ValueError('Frozen visual contract requires Bx3x224x224')
        rgb = image[:, [2, 1, 0]].float()
        tokens = self.backbone(pixel_values=(rgb - self.mean) / self.std).last_hidden_state
        if tuple(tokens.shape[1:]) != (261, 384):
            raise RuntimeError('Unexpected CLS/register/patch token layout')
        return tokens[:, 5:].transpose(1, 2).reshape(-1, 384, 16, 16).contiguous()


class FrozenVisualPrior(nn.Module):
    def __init__(self, model_root, kind='cxformer', random_init=False, seed=7001):
        super().__init__()
        self.encoder = FrozenVisualEncoder(model_root, kind, random_init, seed)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(seed + 1)
            self.project = nn.Sequential(nn.Conv2d(384, 64, 1), nn.GELU(), nn.Conv2d(64, 256, 1))
            nn.init.zeros_(self.project[-1].weight)
            nn.init.zeros_(self.project[-1].bias)
        self.provenance = dict(self.encoder.provenance, adapter_parameters=41280,
            adapter='bilinear_frozen_features_56_then_1x1_384_64_gelu_1x1_64_256',
            insertion='after_down2_before_downVit2', zero_initialized_output=True)

    def forward(self, feature, image):
        if tuple(feature.shape[1:]) != (256, 56, 56):
            raise ValueError('Visual prior must be injected into x3')
        with torch.no_grad():
            frozen = F.interpolate(self.encoder(image), size=(56, 56), mode='bilinear', align_corners=False)
        return feature + self.project(frozen)

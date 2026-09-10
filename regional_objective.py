"""Full-resolution regional overlap objectives; no parameters or extra heads."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from utils import WeightedDiceFocal

MODES = ('global', 'local', 'balanced')


def region_terms(probabilities, targets, mode, window=56, stride=28, eps=1e-6):
    if mode not in MODES:
        raise ValueError('Unknown regional mode')
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    if probabilities.ndim != 4 or probabilities.shape[1] != 1 or targets.shape != probabilities.shape:
        raise ValueError('Expected matched binary [B,1,H,W] tensors')
    if eps <= 0 or not math.isfinite(eps):
        raise ValueError('Positive finite epsilon required')
    p = probabilities
    y = (targets > 0).to(p.dtype)
    if mode == 'global':
        ps, ys, tp = (v.flatten(1).sum(1, keepdim=True) for v in (p, y, p*y))
        area = p.shape[-2]*p.shape[-1]
    else:
        h, w = p.shape[-2:]
        if not 0 < stride <= window <= min(h, w) or (h-window) % stride or (w-window) % stride:
            raise ValueError('Windows must cover every edge without padding')
        area = window*window
        # Fixed avg_pool2d supports deterministic backward in the locked CUDA runtime.
        ps, ys, tp = (F.avg_pool2d(v, window, stride).flatten(1)*area for v in (p, y, p*y))
    positive = ys > 0
    overlap = 1 - (tp+eps)/(ps+ys-tp+eps)
    empty = ps/area
    terms = torch.where(positive, overlap, empty)
    if mode == 'balanced':
        pos_n = positive.sum(1)
        neg_n = (~positive).sum(1)
        pos_mean = (terms*positive).sum(1)/pos_n.clamp_min(1)
        neg_mean = (terms*(~positive)).sum(1)/neg_n.clamp_min(1)
        groups = (pos_n > 0).to(p.dtype)+(neg_n > 0).to(p.dtype)
        per_image = (pos_mean+neg_mean)/groups.clamp_min(1)
    else:
        per_image = terms.mean(1)
    return per_image, terms, positive


class RegionalOverlapObjective(nn.Module):
    def __init__(self, mode, weight, window=56, stride=28, eps=1e-6, **main_kwargs):
        super().__init__()
        if mode not in MODES or not math.isfinite(weight) or weight < 0:
            raise ValueError('Invalid regional objective')
        self.mode, self.weight = mode, float(weight)
        self.window, self.stride, self.eps = window, stride, eps
        self.segmentation = WeightedDiceFocal(**main_kwargs)
        self.last_components = {}

    def region(self, p, y):
        return region_terms(p, y, self.mode, self.window, self.stride, self.eps)[0].mean()

    def _show_dice(self, p, y):
        return self.segmentation._show_dice(p, y)

    def forward(self, p, y):
        main = self.segmentation(p, y)
        if self.weight == 0:
            self.last_components = dict(self.segmentation.last_components)
            return main
        per_image, _, positive = region_terms(p, y, self.mode, self.window, self.stride, self.eps)
        region = per_image.mean()
        weighted = self.weight*region
        total = main+weighted
        self.last_components = dict(main_dice=self.segmentation.last_components['dice'],
            main_focal=self.segmentation.last_components['focal'], main=main.detach(),
            region_raw=region.detach(), region_weighted=weighted.detach(),
            empty_window_fraction=(~positive).float().mean().detach(), total=total.detach())
        return total


def regional_metadata(config):
    if config.regional_mode == 'none':
        return None
    return dict(version='regional_overlap_v1', mode=config.regional_mode, weight=config.regional_weight,
        window=56, stride=28, eps=1e-6, empty_rule='mean_foreground_probability',
        image_aggregation='macro', prediction='final_probability', extra_parameters=0)

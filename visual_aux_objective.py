"""Registered C8 visual terms, without text tasks or report consistency."""
import torch
from torch import nn
from torch.nn import functional as F
from utils import WeightedDiceFocal


def masked_mean(values, valid):
    return (values*valid).sum()/valid.sum().clamp_min(1)


class VisualAuxiliaryObjective(nn.Module):
    def __init__(self,mode,**kwargs):
        super().__init__()
        if mode not in ('pixel','regional'): raise ValueError('Unknown visual supervision mode')
        self.mode=mode
        self.segmentation=WeightedDiceFocal(**kwargs)
        self.last_components={}

    def _show_dice(self,inputs,targets):
        return self.segmentation._show_dice(inputs,targets)

    def components(self,outputs,targets):
        mask=targets.float()
        if mask.ndim==3: mask=mask.unsqueeze(1)
        if mask.ndim!=4 or mask.shape[1]!=1: raise ValueError('Binary mask required')
        pixel,presence,occupancy=[],[],[]
        for route in outputs['pe_routes']:
            gt=F.adaptive_avg_pool2d(mask,route['extent_logits'].shape[-2:])
            pixel.append(F.binary_cross_entropy_with_logits(route['extent_logits'],gt))
            if self.mode=='regional':
                mass=route['basis'].sum((2,3));valid=mass>0
                area=(gt*route['basis']).sum((2,3))/mass.clamp_min(1)
                presence.append(masked_mean(F.binary_cross_entropy_with_logits(
                    route['presence_logits'],(area>0).float(),reduction='none'),valid))
                occupancy.append(masked_mean((route['occupancy']-area).square(),valid))
        losses=dict(main=self.segmentation(outputs['final'],targets),
                    pixel=.02*torch.stack(pixel).mean())
        if self.mode=='regional':
            losses.update(presence=.01*torch.stack(presence).mean(),
                          occupancy=.005*torch.stack(occupancy).mean())
        return losses

    def forward(self,outputs,targets):
        values=self.components(outputs,targets)
        self.last_components={k:v.detach() for k,v in values.items()}
        self.last_components['visual_aux_weighted']=sum(v.detach() for k,v in values.items() if k!='main')
        return sum(values.values())

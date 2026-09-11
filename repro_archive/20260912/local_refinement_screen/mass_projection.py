"""Matched small residual heads and an implicit-gradient soft-mass projection.

Only the residual is trainable; frozen base logits receive no gradient.
All variants share parameter shapes and the exact initial state.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

VARIANTS = ('T1','T2','T3','T4','image','template')
PROJECTED = ('T3','T4','image','template')


class MassProject(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, delta):
        assert not logits.requires_grad, 'This operator assumes a frozen baseline'
        dims = tuple(range(1,logits.ndim))
        base = logits.sigmoid()
        target = base.double().sum(dims,keepdim=True)
        low = delta.amin(dims,keepdim=True)
        high = delta.amax(dims,keepdim=True)
        constant = low == high
        # Float64 tests receive 48 iterations; CUDA float32 uses 24 iterations.
        for _ in range(48 if delta.dtype==torch.float64 else 24):
            mid = (low+high)*.5
            total = (logits+delta-mid).sigmoid().double().sum(dims,keepdim=True)
            low = torch.where(total>target,mid,low)
            high = torch.where(total>target,high,mid)
        out = (logits+delta-(low+high)*.5).sigmoid()
        out = torch.where(constant,base,out)
        ctx.save_for_backward(out); ctx.dims = dims
        return out

    @staticmethod
    def backward(ctx, grad):
        (p,) = ctx.saved_tensors
        w = p*(1-p)
        denominator = w.double().sum(ctx.dims,keepdim=True).clamp_min(1e-30)
        average = (grad*w).double().sum(ctx.dims,keepdim=True)/denominator
        return None, w*(grad-average.to(grad.dtype))


def interpolation_matrix(size=224, source=28):
    x = ((torch.arange(size)+.5)*source/size-.5).clamp(0,source-1)
    low = x.floor().long(); high = (low+1).clamp_max(source-1)
    fraction = x-low
    matrix = torch.zeros(size,source)
    matrix[torch.arange(size),low] += 1-fraction
    matrix[torch.arange(size),high] += fraction
    return matrix


def coordinates(size=28):
    a = torch.linspace(-1,1,size)
    y,x = torch.meshgrid(a,a,indexing='ij')
    return torch.stack([value for k in range(1,17) for value in
        (torch.sin(k*math.pi*x),torch.cos(k*math.pi*x),torch.sin(k*math.pi*y),torch.cos(k*math.pi*y))])[None]


class ResidualHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Conv2d(64,32,1,bias=False)
        self.k = nn.Linear(768,32,bias=False)
        self.v = nn.Linear(768,32,bias=False)
        self.token_offset = nn.Parameter(torch.randn(1,32,768)*.02)
        self.out = nn.Linear(32,1,bias=False)
        nn.init.zeros_(self.out.weight)
        self.register_buffer('position',coordinates())
        self.register_buffer('upsample',interpolation_matrix())

    def evidence(self, query, text, mask):
        tokens = F.layer_norm(text,(768,))+self.token_offset
        key = self.k(tokens); value = self.v(tokens)
        affinity = torch.matmul(query,key.transpose(1,2))/math.sqrt(32)
        affinity = affinity.masked_fill(~mask[:,None].bool(),-1e4)
        context = torch.matmul(affinity.softmax(-1),value)
        return self.out(query*context).transpose(1,2).reshape(-1,1,28,28)

    def forward(self, feature, text, mask, reference, reference_mask, eligible, variant):
        assert variant in VARIANTS
        visual = F.normalize(feature.float(),dim=1)
        spatial = self.position.expand(len(feature),-1,-1,-1)
        input_feature = spatial if variant=='template' else visual+spatial
        query = self.q(input_feature).flatten(2).transpose(1,2)
        if variant=='image':
            # No sample-specific embedding, token length, or mask enters this control.
            text = torch.zeros_like(text); mask = torch.ones_like(mask)
        signal = self.evidence(query,text,mask)
        if variant in ('T2','T4'):
            signal = signal-self.evidence(query,reference,reference_mask)
        residual = .5*signal.tanh()
        full = self.upsample@residual@self.upsample.T
        return full*eligible[:,None,None,None].to(full.dtype)


def prediction(logits, delta, variant):
    return MassProject.apply(logits,delta) if variant in PROJECTED else (logits+delta).sigmoid()

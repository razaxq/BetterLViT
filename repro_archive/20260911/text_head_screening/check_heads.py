"""Behavioral CPU/CUDA preflight; no real image or mask evaluation."""
import argparse
import copy
import json
import time
import torch
from torch.nn import functional as F
from heads import ResidualHead, MassProject, prediction, interpolation_matrix, VARIANTS, PROJECTED


def run(device):
    torch.manual_seed(1219); torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # Independent finite differences validate the implicit, coupled gradient.
    z = torch.randn(2,1,3,4,dtype=torch.float64)
    d = (torch.randn_like(z)*.1).requires_grad_()
    assert torch.autograd.gradcheck(MassProject.apply,(z,d),eps=1e-5,atol=2e-6,rtol=2e-4)
    p = MassProject.apply(z,d)
    assert (p.sum((1,2,3))-z.sigmoid().sum((1,2,3))).abs().max()<1e-10
    assert torch.equal(MassProject.apply(z,torch.zeros_like(z)),z.sigmoid())
    assert torch.equal(MassProject.apply(z,torch.ones_like(z)*.2),z.sigmoid())
    assert torch.allclose(p,MassProject.apply(z,d+.2),atol=1e-12)
    extreme = torch.tensor([[-100.,100.],[-20.,20.]],dtype=torch.float64).reshape(1,1,2,2)
    de = torch.randn_like(extreme,requires_grad=True)
    pe = MassProject.apply(extreme,de);pe.square().sum().backward()
    assert torch.isfinite(pe).all() and torch.isfinite(de.grad).all()
    w = interpolation_matrix(); low = torch.randn(2,1,28,28)
    assert torch.allclose(w@low@w.T,F.interpolate(low,size=(224,224),mode='bilinear',align_corners=False),atol=1e-6)
    head = ResidualHead().to(device)
    f = torch.randn(2,64,28,28,device=device)
    text = torch.randn(2,32,768,device=device); ref=text.flip(1)
    mask = torch.ones(2,32,device=device,dtype=torch.bool); mask[:,24:]=False
    eligible=torch.ones(2,device=device,dtype=torch.bool)
    logits=torch.randn(2,1,224,224,device=device)*5
    for variant in VARIANTS:
        delta=head(f,text,mask,ref,mask,eligible,variant)
        assert torch.equal(prediction(logits,delta,variant),logits.sigmoid()),variant
    with torch.no_grad(): head.out.weight.normal_(std=.1)
    losses={}; maximum_mass_error=0.
    for variant in VARIANTS:
        head.zero_grad(set_to_none=True)
        delta=head(f,text,mask,ref,mask,eligible,variant)
        p=prediction(logits,delta,variant)
        assert delta.abs().max()<=.5 and torch.isfinite(p).all()
        again=prediction(logits,head(f,text,mask,ref,mask,eligible,variant),variant)
        assert torch.equal(p,again),variant
        if variant in PROJECTED:
            error=float((p.double().sum((1,2,3))-logits.sigmoid().double().sum((1,2,3))).abs().max())
            maximum_mass_error=max(maximum_mass_error,error); assert error<=.02,error
        loss=F.binary_cross_entropy(p.clamp(1e-6,1-1e-6),(logits>0).float())
        loss.backward(); losses[variant]=float(loss.detach())
        for name,param in head.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(),(variant,name)
        assert float(head.k.weight.grad.abs().sum())>0
        assert torch.equal(head(f,text,mask,ref,mask,eligible*False,variant),torch.zeros_like(logits))
    a=head(f,text,mask,ref,mask,eligible,'image')
    b=head(f,text*100,~mask,ref*100,~mask,eligible,'image')
    assert torch.equal(a,b),'Image control leaked text'
    assert torch.equal(head(f,text,mask,ref,mask,eligible,'template'),head(f*100,text,mask,ref,mask,eligible,'template'))
    assert torch.equal(head(f,text,mask,text,mask,eligible,'T4'),torch.zeros_like(logits))
    # A real backward repeat must agree, not only repeat inference.
    gradients=[]
    for _ in range(2):
        head.zero_grad(set_to_none=True)
        prediction(logits,head(f,text,mask,ref,mask,eligible,'T4'),'T4').square().mean().backward()
        gradients.append(torch.cat([p.grad.flatten() for p in head.parameters()]))
    assert torch.equal(*gradients)
    return dict(passed=True,device=device,parameters=sum(p.numel() for p in head.parameters()),
        finite_difference_gradient=True,identity=True,mass_max_error_pixels=maximum_mass_error,
        deterministic_forward_backward=True,image_text_invariance=True,template_image_invariance=True,
        equal_reference_zero=True,eligibility_abstention=True,all_parameter_gradients_finite=True,
        synthetic_only=True,losses=losses)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');a=p.parse_args()
    print(json.dumps(run(a.device)))

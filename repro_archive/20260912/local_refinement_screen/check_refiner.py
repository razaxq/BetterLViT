"""Synthetic behavioral and differentiation tests for the new local interface."""
import copy
import json
import torch
from refiner import LocalRefiner,VARIANTS,predict
from mass_projection import MassProject

def run(device):
    torch.set_num_threads(4);torch.manual_seed(1219);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    zsmall=torch.randn(2,7,dtype=torch.float64);dsmall=(torch.randn_like(zsmall)*.1).requires_grad_()
    assert torch.autograd.gradcheck(MassProject.apply,(zsmall,dsmall),eps=1e-5,atol=2e-6,rtol=2e-4)
    head=LocalRefiner().to(device);z=torch.randn(2,1,224,224,device=device)*3
    coarse=torch.randn(2,64,28,28,device=device);fine=torch.randn(2,1024,64,device=device)
    ix=torch.stack([torch.randperm(224*224,device=device)[:1024] for _ in range(2)])
    selected=torch.zeros_like(z,dtype=torch.bool).flatten(1).scatter(1,ix,True).reshape_as(z)
    inputs=(coarse,fine,z,ix)
    for v in VARIANTS:
        assert torch.equal(predict(z,head(*inputs,v),ix,v),z.sigmoid()),v
    with torch.no_grad():head.out.weight.normal_(std=.1)
    errors={}
    for v in VARIANTS:
        grads=[]
        for _ in range(2):
            head.zero_grad(set_to_none=True);delta=head(*inputs,v);p=predict(z,delta,ix,v)
            assert float(delta.abs().max())<=.5 and torch.isfinite(p).all()
            assert torch.equal(p[~selected],z.sigmoid()[~selected])
            p.square().mean().backward()
            assert all(param.grad is not None and torch.isfinite(param.grad).all() for param in head.parameters())
            grads.append(torch.cat([param.grad.flatten() for param in head.parameters()]))
        assert torch.equal(*grads),v
        if v.endswith('mass'):
            errors[v]=float((p.double().sum((1,2,3))-z.sigmoid().double().sum((1,2,3))).abs().max())
            assert errors[v]<=.02
        assert float(head.fc1.weight.grad.abs().sum())>0,v
    return dict(passed=True,device=device,parameters=sum(p.numel() for p in head.parameters()),
        local_identity=True,outside_candidates_exact=True,projection_gradcheck=True,
        deterministic_forward_backward=True,finite_gradients=True,mass_errors=errors)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');a=p.parse_args();print(json.dumps(run(a.device)))

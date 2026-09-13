"""Behavioral regressions for visual-only values and both normalization axes."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets.decoder_context import VisualValueRegroupAdapter, DecoderContextAdapter, adapter_metadata

def main():
    torch.set_num_threads(2);torch.manual_seed(1219)
    x=torch.randn(3,128,8,16);text=torch.randn(3,32,128)
    mask=torch.arange(32)[None,:]<torch.tensor([15,2,0])[:,None]
    initial=torch.get_rng_state().clone();modules=[]
    for mode in ('regroup_visual','regroup_text'):
        with torch.random.fork_rng(devices=[]):m=VisualValueRegroupAdapter(mode)
        assert torch.equal(initial,torch.get_rng_state())
        assert sum(p.numel() for p in m.parameters())==16896
        assert torch.equal(m(x,text,mask),x)
        modules.append(m)
    assert all(torch.equal(v,modules[1].state_dict()[k]) for k,v in modules[0].state_dict().items())
    checks={}
    for m in modules:
        with torch.no_grad():m.output.weight.normal_(0,.05)
        y=m(x,text,mask)
        if m.mode=='text':
            assert torch.equal(y[1:],x[1:])
            dirty=text.masked_fill(~mask[:,:,None],10000.)
            assert torch.equal(y,m(x,dirty,mask))
            assert not torch.equal(y[0],m(x,text.flip(0),mask)[0])
        else:
            assert torch.equal(y,m(x,text.flip(0),~mask))
        flat=x[:,:,:1,:1].expand_as(x).contiguous()
        err=float((m(flat,text,mask)-flat).abs().max());assert err<2e-6
        # With nonzero output projection, zero image produces zero update for
        # arbitrary report values. Original T2 does not have this invariant.
        assert torch.count_nonzero(m(torch.zeros_like(x),text,mask))==0
        captured=[]
        hook=m.value.register_forward_pre_hook(lambda module,inputs:captured.append(inputs[0].detach().clone()))
        m(x,text,mask);hook.remove()
        assert torch.equal(captured[0],m.query_norm(x.flatten(2).transpose(1,2)))
        grads=[];m=VisualValueRegroupAdapter(m.regroup_mode)
        opt=torch.optim.Adam(m.parameters(),lr=.001)
        for i in range(3):
            opt.zero_grad(set_to_none=True);out=m(x,text,mask);(out-1).square().mean().backward()
            g={n:float(p.grad.abs().max()) for n,p in m.named_parameters()}
            assert all(torch.isfinite(p.grad).all() for p in m.parameters())
            assert g['output.weight']>0
            if i:assert all(g[n+'.weight']>0 for n in ('query','key','value'))
            opt.step();grads.append(g)
        m.capture_stats=False;a=m(x,text,mask)
        m.capture_stats=True;b=m(x,text,mask);assert torch.equal(a,b)
        checks[m.regroup_mode]=dict(constant_image_max_error=err,gradients=grads)
    print(json.dumps(dict(status='ok',parameters=16896,initialization_and_rng_exact=True,
        masking_and_empty=True,visual_values_only=True,zero_image_no_injection=True,
        constant_image_no_injection=True,observation_forward_exact=True,checks=checks)))

if __name__=='__main__':main()

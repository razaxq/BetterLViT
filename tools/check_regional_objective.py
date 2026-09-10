"""Independent geometry, empty-region, macro reduction and gradient regressions."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from regional_objective import region_terms,RegionalOverlapObjective,MODES
from utils import WeightedDiceFocal


def main():
    torch.manual_seed(13)
    p=torch.rand(2,1,8,8,dtype=torch.float64,requires_grad=True)
    y=torch.zeros_like(p);y[0,0,1:3,1:3]=1;y[1]=1
    for mode in MODES:
        actual,terms,positive=region_terms(p,y,mode,4,2)
        reference=[]
        for b in range(2):
            values=[];flags=[]
            coords=[(0,0,8)] if mode=='global' else [(i,j,4) for i in range(0,5,2) for j in range(0,5,2)]
            for i,j,k in coords:
                x=p[b,0,i:i+k,j:j+k];g=y[b,0,i:i+k,j:j+k]
                present=bool(g.sum()>0);flags.append(present)
                values.append(1-((x*g).sum()+1e-6)/(x.sum()+g.sum()-(x*g).sum()+1e-6) if present else x.mean())
            values=torch.stack(values);flags=torch.tensor(flags)
            if mode=='balanced':
                groups=[values[mask].mean() for mask in (flags,~flags) if mask.any()]
                reference.append(torch.stack(groups).mean())
            else:reference.append(values.mean())
        expected=torch.stack(reference)
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        a=torch.autograd.grad(actual.mean(),p,retain_graph=True)[0]
        b=torch.autograd.grad(expected.mean(),p,retain_graph=True)[0]
        torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-12)
        flipped=region_terms(p.flip(-1),y.flip(-1),mode,4,2)[0]
        torch.testing.assert_close(actual,flipped,rtol=1e-12,atol=1e-12)
        empty=region_terms(p,torch.zeros_like(p),mode,4,2)[0].mean()
        assert (torch.autograd.grad(empty,p,retain_graph=True)[0]>0).all()
        full=region_terms(p,torch.ones_like(p),mode,4,2)[0].mean()
        assert (torch.autograd.grad(full,p,retain_graph=True)[0]<0).all()
        one=torch.zeros_like(y);one[0,0,0,0]=1
        v=region_terms(p,one,mode,4,2)[0].mean()
        assert torch.isfinite(torch.autograd.grad(v,p,retain_graph=True)[0]).all()
        disabled=RegionalOverlapObjective(mode,0)(p,y)
        original=WeightedDiceFocal()(p,y)
        assert torch.equal(disabled,original)
        torch.testing.assert_close(torch.autograd.grad(disabled,p,retain_graph=True)[0],
                                   torch.autograd.grad(original,p,retain_graph=True)[0],rtol=0,atol=0)
    good=torch.zeros(1,1,8,8);good[:,:,0:2,0:2]=1
    bad=good.roll((5,5),(-2,-1))
    assert region_terms(good,good,'global')[0].item()==0
    assert region_terms(bad,good,'global')[0].item()>.99
    try:region_terms(p,y,'local',5,2)
    except ValueError:pass
    else:raise AssertionError('Incomplete edge coverage accepted')
    print(json.dumps(dict(status='ok',checks=['independent_crop_values','independent_crop_gradients',
        'per_image_macro','group_reduction','flip_geometry','empty_fp_gradient','full_fn_gradient',
        'single_pixel_finite','disabled_exact','same_area_displacement','edge_coverage'])))


if __name__=='__main__':main()

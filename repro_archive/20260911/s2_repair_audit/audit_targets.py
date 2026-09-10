"""Read-only Train mask audit: cross-scale regional targets and mean-pool dilution."""
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np
from scipy import ndimage
import torch
from torch.nn import functional as F

REPO=Path('/root/autodl-tmp/BetterLViT-visual-aux-s2')
SOURCE='7defc637e36974fabd0f47763275c90f617e5f53'
sys.path.insert(0,str(REPO))
from race_semantics import make_zone_basis


def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==SOURCE
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=REPO,text=True).strip()
    torch.set_num_threads(2)
    folder=REPO/'datasets/Covid19/Train_Folder/labelcol'
    files=sorted(folder.iterdir());assert len(files)==5716
    base=make_zone_basis(224,224);py=random.Random(1219);rng=np.random.RandomState(1219)
    stats={v:{str(s):dict(valid=0,positive_reference=0,flips=0,false_positive_target=0,false_negative_target=0,area_abs_error_sum=0.,affected_images=0) for s in (224,112,56,28)} for v in ('native','one_legacy_augmentation')}
    fractions={v:[] for v in stats};examples=[]
    @torch.no_grad()
    def batch(masks,bases,names,variant):
        m=torch.from_numpy(np.stack(masks)).float().unsqueeze(1);z=torch.from_numpy(np.stack(bases)).float()
        mass=z.sum((2,3));valid=mass>0
        ref=(m*z).sum((2,3))/mass.clamp_min(1);present=ref>0
        fractions[variant].extend(ref[present&valid].tolist())
        for size in (224,112,56,28):
            gt=F.adaptive_avg_pool2d(m,(size,size));basis=F.interpolate(z,(size,size),mode='nearest')
            old_mass=basis.sum((2,3));old_valid=old_mass>0
            area=(gt*basis).sum((2,3))/old_mass.clamp_min(1);old=area>0;both=valid&old_valid
            flips=(old!=present)&both;row=stats[variant][str(size)]
            row['valid']+=int(both.sum());row['positive_reference']+=int((present&both).sum())
            row['flips']+=int(flips.sum());row['false_positive_target']+=int((old&~present&both).sum())
            row['false_negative_target']+=int((~old&present&both).sum());row['affected_images']+=int(flips.any(1).sum())
            row['area_abs_error_sum']+=float((area-ref).abs()[both].sum())
            if size==28 and len(examples)<16:
                for i,j in flips.nonzero().tolist():
                    if len(examples)==16:break
                    examples.append(dict(variant=variant,name=names[i],zone=j,reference_area=float(ref[i,j]),coarse_area=float(area[i,j])))
    pending={v:([],[],[]) for v in stats}
    for path in files:
        mask=cv2.imread(str(path),0);assert mask is not None
        mask=(cv2.resize(mask,(224,224))>0).astype(np.uint8)
        augmented=mask.copy();basis=base.copy()
        if py.random()>.5:
            k=rng.randint(0,4);axis=rng.randint(0,2)
            augmented=np.flip(np.rot90(augmented,k),axis=axis).copy()
            basis=np.flip(np.rot90(basis,k,axes=(1,2)),axis=axis+1).copy()
        elif py.random()>.5:
            angle=rng.randint(-20,20)
            augmented=ndimage.rotate(augmented,angle,order=0,reshape=False)
            basis=ndimage.rotate(basis,angle,axes=(1,2),order=0,reshape=False)
        for variant,m,z in [('native',mask,base),('one_legacy_augmentation',augmented,basis)]:
            mm,zz,nn=pending[variant];mm.append(m);zz.append(z);nn.append(path.name)
            if len(mm)==32:batch(mm,zz,nn,variant);mm.clear();zz.clear();nn.clear()
    for variant,(mm,zz,nn) in pending.items():
        if mm:batch(mm,zz,nn,variant)
    for variant in stats:
        for row in stats[variant].values():
            row['flip_fraction']=row['flips']/row['valid'];row['area_mae']=row['area_abs_error_sum']/row['valid']
    dilution={v:dict(positive_regions=len(x),foreground_fraction_quantiles=np.quantile(x,[.1,.25,.5,.75,.9]).tolist(),
        positive_regions_below_1_percent=sum(t<.01 for t in x),positive_regions_below_5_percent=sum(t<.05 for t in x)) for v,x in fractions.items()}
    result=dict(source_git_commit=SOURCE,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Train labels only; all 5716 masks; native and one reproducible legacy transform per mask; no model or optimizer',
        dataset_count=5716,dataset_names_sha256=hashlib.sha256('\n'.join(p.name for p in files).encode()).hexdigest(),
        test_split_accessed=False,validation_split_accessed=False,training_performed=False,stats=stats,dilution=dilution,examples=examples,
        torch_version=torch.__version__,numpy_version=np.__version__,opencv_version=cv2.__version__)
    print(json.dumps(result))


if __name__=='__main__':main()

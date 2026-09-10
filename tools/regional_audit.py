"""Fixed Train-only output-gradient audit, preserving all training state."""
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import Config as config
from Load_Dataset import ImageToImage2D, ValGenerator
from utils import read_text, WeightedDiceFocal
from regional_objective import region_terms, MODES


def digest(items):
    h=hashlib.sha256()
    for name,value in items:
        h.update(name.encode())
        if value is not None:
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def fixed_loader():
    data=ImageToImage2D(config.train_dataset,config.task_name,
        read_text(os.path.join(config.task_dataset,'Train_Val_text.xlsx')),
        ValGenerator([224,224]),image_size=224)
    indices=sorted(range(len(data)),key=lambda i:hashlib.sha256(
        ('regional-overlap-v1:'+data.mask_list[i]).encode()).hexdigest())[:32]
    return DataLoader(Subset(data,indices),batch_size=8,shuffle=False,num_workers=0)


def measure(model, modes=MODES, weight=1.0):
    records=[]
    for sample,names in fixed_loader():
        sample={k:v.cuda() for k,v in sample.items()}
        with torch.no_grad():
            p=model(sample['image'],sample['input_ids'],sample['attention_mask'])
        p=p.detach().requires_grad_(True)
        y=sample['label'].float()
        main=WeightedDiceFocal()(p,y)
        base=torch.autograd.grad(main,p)[0]*p.detach()*(1-p.detach())
        base_norm=base.norm().item()
        stats={}
        for mode in modes:
            per_image,terms,pos=region_terms(p,y,mode)
            loss=per_image.mean()
            gradient=torch.autograd.grad(loss,p)[0]*p.detach()*(1-p.detach())
            norm=gradient.norm().item()
            stats[mode]=dict(raw_loss=loss.item(),weighted_loss=weight*loss.item(),
                raw_logit_gradient_ratio=norm/base_norm,
                weighted_logit_gradient_ratio=weight*norm/base_norm,
                logit_gradient_cosine=(gradient*base).sum().item()/(norm*base_norm),
                empty_window_fraction=(~pos).float().mean().item())
        pred=p.detach()[:,0]>.5;gt=y>0
        if gt.ndim==4:gt=gt[:,0]
        cases=[]
        for i,name in enumerate(names):
            tp=int((pred[i]&gt[i]).sum());fp=int((pred[i]&~gt[i]).sum());fn=int((~pred[i]&gt[i]).sum())
            cases.append(dict(name=name,tp=tp,fp=fp,fn=fn,label_pixels=tp+fn))
        records.append(dict(cases=cases,main_loss=main.item(),gradients=stats))
    return records


def run_audit(model,criterion,epoch,directory):
    path=Path(directory)/f'epoch_{epoch:03d}.json'
    assert not path.exists()
    target=model.module if hasattr(model,'module') else model
    modes={n:m.training for n,m in target.named_modules()}
    python_state=random.getstate();numpy_state=np.random.get_state()
    before=digest(target.state_dict().items())
    grad_before=digest((n,p.grad) for n,p in target.named_parameters())
    components=dict(criterion.last_components)
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            target.eval()
            records=measure(target,(criterion.mode,),criterion.weight)
    finally:
        for name,module in target.named_modules():module.training=modes[name]
        random.setstate(python_state);np.random.set_state(numpy_state)
        criterion.last_components=components
    after=digest(target.state_dict().items())
    assert before==after and grad_before==digest((n,p.grad) for n,p in target.named_parameters())
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(epoch=epoch,source_git_commit=config.source_git_commit,
        mode=criterion.mode,weight=criterion.weight,scope='32 fixed Train images; final-logit gradients, not parameter gradients',
        state_sha256_before=before,state_sha256_after=after,gradient_state_unchanged=True,
        rng_restored=True,batches=records),indent=2)+'\n')

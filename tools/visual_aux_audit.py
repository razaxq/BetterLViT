"""Scheduled Train-only telemetry that preserves model, gradient and RNG state."""
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from Load_Dataset import ImageToImage2D, ValGenerator
from utils import read_text
import Config as config

_loader=None


def digest(items):
    h=hashlib.sha256()
    for name,value in items:
        h.update(name.encode())
        if value is not None: h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run_audit(model, criterion, epoch, directory):
    global _loader
    destination=Path(directory)/f'epoch_{epoch:03d}.json'
    assert not destination.exists()
    target=model.module if hasattr(model,'module') else model
    modes={name:m.training for name,m in target.named_modules()}
    python_state=random.getstate();numpy_state=np.random.get_state()
    before=digest(target.state_dict().items())
    grad_before=digest((n,p.grad) for n,p in target.named_parameters())
    features={};hooks=[];records=[]
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            target.eval()
            if _loader is None:
                data=ImageToImage2D(config.train_dataset,config.task_name,
                    read_text(os.path.join(config.task_dataset,'Train_Val_text.xlsx')),
                    ValGenerator([224,224]),image_size=224)
                indices=sorted(range(len(data)),key=lambda i:hashlib.sha256(
                    ('r2-visual-aux-v1:'+data.mask_list[i]).encode()).hexdigest())[:32]
                _loader=DataLoader(Subset(data,indices),batch_size=8,shuffle=False,num_workers=0)
            for name in ('inc','down1','down2','down3'):
                hooks.append(getattr(target,name).register_forward_hook(
                    lambda m,i,o,name=name:features.__setitem__(name,o)))
            for sample,names in _loader:
                sample={k:v.cuda() for k,v in sample.items()}
                outputs=target(sample['image'],sample['input_ids'],sample['attention_mask'],
                    return_aux=True,race_slot_targets=sample['race_slot_targets'],
                    race_zone_basis=sample['race_zone_basis'])
                taps=[features[k] for k in ('inc','down1','down2','down3')]
                losses=criterion.components(outputs,sample['label'].float())
                losses['visual_aux']=sum(v for k,v in losses.items() if k!='main')
                gradients={k:torch.autograd.grad(v,taps,retain_graph=True) for k,v in losses.items()}
                stats={}
                for name,gs in gradients.items():
                    stats[name]=[]
                    for g,ref in zip(gs,gradients['main']):
                        norm=g.norm().item();base=ref.norm().item()
                        stats[name].append(dict(ratio=norm/base if base else None,
                            cosine=(g*ref).sum().item()/(norm*base) if norm*base else None))
                pred=outputs['final'].detach()[:,0]>.5
                gt=sample['label']>0
                if gt.ndim==4:gt=gt[:,0]
                cases=[]
                for i,name in enumerate(names):
                    tp=int((pred[i]&gt[i]).sum());fp=int((pred[i]&~gt[i]).sum());fn=int((~pred[i]&gt[i]).sum())
                    cases.append(dict(name=name,tp=tp,fp=fp,fn=fn,label_pixels=tp+fn,prediction_pixels=tp+fp))
                records.append(dict(cases=cases,gradients=stats,weighted_losses={k:v.item() for k,v in losses.items()}))
                features.clear()
                del outputs,taps,gradients,losses,sample
    finally:
        for hook in hooks:hook.remove()
        features.clear()
        for name,module in target.named_modules():module.training=modes[name]
        random.setstate(python_state);np.random.set_state(numpy_state)
    after=digest(target.state_dict().items())
    grad_after=digest((n,p.grad) for n,p in target.named_parameters())
    assert before==after and grad_before==grad_after
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(dict(epoch=epoch,source_git_commit=config.source_git_commit,
        mode=config.visual_aux_mode,scope='32 fixed Train images; eval; no optimizer updates',
        state_sha256_before=before,state_sha256_after=after,gradient_state_unchanged=True,
        rng_restored=True,batches=records),indent=2)+'\n')

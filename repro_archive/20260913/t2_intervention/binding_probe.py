"""Separate relation-sensitive behavior from generic same-length mismatch.

Also measures local upstream-gradient contributions on four fixed Train batches.
No optimizer steps, no generated masks, no Test reads.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np
import torch
from diagnose import digest,model_digest,records,EXPECTED

class BypassGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx,feature,output):
        return output
    @staticmethod
    def backward(ctx,gradient):
        return gradient,None

def binding(text):
    clauses=text.lower().strip().rstrip('.').split(',')
    if len(clauses)!=3:return None
    slots={}
    for clause in clauses[-1].strip().split(' and '):
        m=re.fullmatch(r'((?:upper|middle|lower|all)(?:\s+(?:upper|middle|lower|all))*)\s+(left|right)\s+lung',clause.strip())
        if not m or m[2] in slots:return None
        words=set(m[1].split())
        slots[m[2]]=tuple(sorted({'upper','middle','lower'} if 'all' in words else words))
    return tuple(sorted(slots.items()))

def match_binding(dataset):
    names=[n.replace('mask_','') for n in dataset.mask_list]
    texts=[dataset.rowtext[n] for n in dataset.mask_list]
    relations=[binding(t) for t in texts]
    buckets=defaultdict(list)
    for i,(ids,mask) in enumerate(zip(dataset.input_ids,dataset.attention_masks)):
        buckets[tuple(sorted(ids[mask.bool()].tolist()))].append(i)
    donors=list(range(len(names)))
    for bucket in buckets.values():
        ordered=sorted(bucket,key=lambda i:names[i])
        for pos,i in enumerate(ordered):
            if relations[i] is None:continue
            for shift in range(1,len(ordered)):
                j=ordered[(pos+shift)%len(ordered)]
                if relations[j] is not None and relations[i]!=relations[j] and not torch.equal(dataset.input_ids[i],dataset.input_ids[j]):
                    donors[i]=j;break
    mapping=[dict(name=n,donor=names[donors[i]],changed=donors[i]!=i,
        relation=relations[i],donor_relation=relations[donors[i]]) for i,n in enumerate(names)]
    return names,donors,mapping

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repository',required=True);ap.add_argument('--reference',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();sys.argv=[sys.argv[0]];start=time.time()
    root=Path(a.repository);sys.path[:0]=[str(root),str(root/'tools')]
    import export_validation_metrics as ex
    from Load_Dataset import ImageToImage2D,ValGenerator
    from utils import read_text,WeightedDiceFocal
    from torch.utils.data import DataLoader
    assert ex.git_commit()==EXPECTED
    assert not subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=no'],text=True).strip()
    ref=json.loads(Path(a.reference).read_text());prior={r['name']:r for r in ref['records']}
    ck=torch.load(ref['checkpoint'],map_location='cpu',weights_only=True)
    assert ck['source_git_commit']==EXPECTED and not ck['text_use_lora'] and ck.get('boundary_loss_weight',0)==0
    torch.manual_seed(1219);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    model=ex.build_model();model.load_state_dict(ck['state_dict'],strict=True);del ck
    model=model.cuda().eval();before=model_digest(model)
    dataset,loader=ex.validation_loader(16);names,donors,mapping=match_binding(dataset)
    lookup={n:i for i,n in enumerate(names)};cache=[];out=[]
    with torch.inference_mode():
        for i in range(0,len(names),16):
            t=model.encode_text(dataset.input_ids[i:i+16].cuda(),dataset.attention_masks[i:i+16].cuda())
            for module in (model.text_module4,model.text_module3,model.text_module2):t=module(t.transpose(1,2)).transpose(1,2)
            cache.append(t.cpu())
        cache=torch.cat(cache)
        for bi,(batch,bn) in enumerate(loader):
            ix=[donors[lookup[n]] for n in bn];dt=cache[ix].cuda();dm=dataset.attention_masks[ix].cuda()
            hook=model.decoder_context.register_forward_pre_hook(lambda module,inputs:(inputs[0],dt,dm))
            p=model(batch['image'].cuda(),batch['input_ids'].cuda(),batch['attention_mask'].cuda());hook.remove()
            rows=records(p,batch['label'],bn);out.extend(rows)
            for r in rows:
                if not mapping[lookup[r['name']]]['changed']:
                    old=prior[r['name']];assert r['prediction_pixels']==old['prediction_pixels'] and abs(r['iou']-old['iou'])<1e-12
            if bi%30==0:print(json.dumps(dict(phase='binding',samples=len(out),elapsed_seconds=time.time()-start)),flush=True)
    # Keep the trained model in eval to measure deterministic local sensitivity,
    # not gradients from a new training recipe. All stored weights stay fixed.
    for p in model.parameters():p.requires_grad_(False)
    td=ImageToImage2D(ex.config.train_dataset,ex.config.task_name,
        read_text(os.path.join(ex.config.task_dataset,'Train_Val_text.xlsx')),
        ValGenerator([224,224]),image_size=224)
    dl=DataLoader(td,batch_size=16,shuffle=False,num_workers=0)
    objective=WeightedDiceFocal();gradient_records=[]
    for bi,(batch,bn) in enumerate(dl):
        if bi==4:break
        images,ids,mask,target=[batch[k].cuda() for k in ('image','input_ids','attention_mask','label')]
        gs={};ps={}
        for mode in ('full','stop_residual_gradient'):
            captured={}
            def leaf(module,inputs):
                x=inputs[0].detach().requires_grad_(True);captured['x']=x
                return (x,*inputs[1:])
            pre=model.decoder_context.register_forward_pre_hook(leaf)
            post=None
            if mode=='stop_residual_gradient':
                post=model.decoder_context.register_forward_hook(lambda module,inputs,value:BypassGradient.apply(inputs[0],value))
            prob=model(images,ids,mask);loss=objective(prob,target)
            gs[mode]=torch.autograd.grad(loss,captured['x'])[0].detach()
            ps[mode]=prob.detach()
            pre.remove()
            if post is not None:post.remove()
        assert torch.equal(ps['full'],ps['stop_residual_gradient'])
        g0=gs['stop_residual_gradient'];gr=gs['full']-g0
        dot=(g0*gr).sum((1,2,3));n0=g0.square().sum((1,2,3)).sqrt();nr=gr.square().sum((1,2,3)).sqrt()
        cosine=dot/(n0*nr).clamp_min(1e-30)
        for i,n in enumerate(bn):gradient_records.append(dict(name=n,cosine_residual_vs_bypass=float(cosine[i]),
            residual_gradient_norm_over_bypass=float(nr[i]/n0[i].clamp_min(1e-30)),
            negative_dot=bool(dot[i]<0),label_pixels=int(target[i].bool().sum())))
        print(json.dumps(dict(phase='train_gradient',batch=bi+1,forward_absmax=float((ps['full']-ps['stop_residual_gradient']).abs().max()))),flush=True)
    assert model_digest(model)==before
    result=dict(status='complete',source_git_commit=EXPECTED,analysis_git_commit=os.environ['ANALYSIS_GIT_COMMIT'],
        scripts_sha256={n:digest(Path(__file__).parent/n) for n in ('binding_probe.py','diagnose.py')},
        model_state_unchanged=True,training_performed=False,optimizer_steps=0,test_split_accessed=False,
        validation_samples=len(names),eligible_binding=sum(r['changed'] for r in mapping),mapping=mapping,records=out,
        train_gradient_records=gradient_records,elapsed_seconds=time.time()-start,
        interpretation='Same-wordpiece-bag different-location-binding intervention plus deterministic local Train sensitivity; not training optimization proof')
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='complete',eligible=result['eligible_binding'],elapsed_seconds=result['elapsed_seconds'])),flush=True)

if __name__=='__main__':main()

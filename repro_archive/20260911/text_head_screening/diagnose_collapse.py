"""Post-hoc read-only analysis of saved heads and one original FIT batch.

No optimizer step, official Val/Test access, holdout prediction, or run-status poll.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path('/root/text_head_b_49905dbb')
sys.path[:0]=[str(ROOT),'/root/BetterLViT-recipe-r2']
from heads import ResidualHead,VARIANTS,prediction
from analysis import digest,write_json
from utils import WeightedDiceFocal


def state_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    torch.set_num_threads(4);torch.manual_seed(1219);torch.use_deterministic_algorithms(True)
    m=json.loads((ROOT/'manifest.json').read_text())
    rows=json.loads((ROOT/'split.json').read_text())['records']
    order=json.loads((ROOT/'results/training_order.json').read_text())
    cache=json.loads((ROOT/'results/cache_manifest.json').read_text())
    checkpoint=torch.load(ROOT/'results/heads.pt',map_location='cpu',weights_only=True)
    assert checkpoint['source_git_commit']=='49905dbbdc644454a37a4a49098db0d5df5fe75a'
    assert checkpoint['steps']==1024 and checkpoint['manifest']==m
    assert digest(ROOT/'results/heads.pt')=='3bf949edf52225cb75910d262e3a1df139f3f25b1661e5ec29155c8d3a4abcc2'
    initial=ResidualHead()
    assert state_hash(initial)==order['initial_state_sha256']
    ix=order['indices'][:16]
    assert all(rows[i]['partition']=='fit' and rows[i]['eligible'] for i in ix)
    arrays={key:np.memmap(ROOT/'cache'/(key+'.bin'),dtype=dtype,mode='r',shape=tuple(shape)) for key,(shape,dtype) in cache['shapes'].items()}
    feature=torch.from_numpy(np.array(arrays['features'][ix])).float()
    z=torch.from_numpy(np.array(arrays['logits'][ix]))
    y=torch.from_numpy(np.unpackbits(np.array(arrays['masks'][ix]),axis=1,count=224*224,bitorder='little').reshape(-1,1,224,224)).float()
    text=np.load(ROOT/'cache/text.npz'); embeddings=torch.from_numpy(text['embeddings']); masks=torch.from_numpy(text['masks'])
    lookup={t:i for i,t in enumerate(cache['unique_texts'])}
    correct=[lookup[rows[i]['canonical']] for i in ix];ref=[lookup[rows[i]['reference']] for i in ix]
    inputs=(feature,embeddings[correct],masks[correct],embeddings[ref],masks[ref],torch.ones(16,dtype=torch.bool))
    result=dict(source_git_commit=checkpoint['source_git_commit'],kind='read_only_saved_head_and_train_gradient_diagnosis',
        checkpoint_load_and_hash_verified=True,initial_state_hash_exact=True,
        train_indices=ix,train_names=[rows[i]['name'] for i in ix],model_updates=False,
        internal_holdout_evaluated=False,official_validation_accessed=False,test_split_accessed=False,
        variants={})
    loss_function=WeightedDiceFocal()
    for variant in VARIANTS:
        states={}
        for stage,state in [('initial',initial.state_dict()),('final',checkpoint['heads'][variant])]:
            head=ResidualHead();head.load_state_dict(state)
            delta=head(*inputs,variant);loss=loss_function(prediction(z,delta,variant),y)
            loss.backward()
            parameters={}
            for name,p in head.named_parameters():
                task=p.grad.detach();wd=p.detach()*m['weight_decay']
                pn=float(p.detach().norm());gn=float(task.norm());wn=float(wd.norm())
                parameters[name]=dict(parameter_norm=pn,task_gradient_norm=gn,coupled_l2_gradient_norm=wn,
                    l2_to_task_norm_ratio=wn/max(gn,1e-30),nonzero_parameters=int(p.detach().count_nonzero()),
                    task_l2_cosine=float((task*wd).sum())/max(gn*wn,1e-30))
            states[stage]=dict(loss=float(loss.detach()),mean_abs_delta=float(delta.detach().abs().mean()),
                state_sha256=state_hash(head),parameters=parameters)
        result['variants'][variant]=states
    write_json(ROOT/'posthoc_collapse.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()

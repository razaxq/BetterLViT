"""Real-Train deterministic preflight, legacy/grouped optimizer, and calibration."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch.utils.data import DataLoader
import Config as config
from Load_Dataset import ImageToImage2D,RandomGenerator
from nets.BetterLViT import BetterLViT
from train_model import build_optimizer_parameter_groups
from utils import read_text,WeightedDiceFocal
from regional_objective import RegionalOverlapObjective
from tools.regional_audit import digest,measure,run_audit


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--calibrate',action='store_true')
    parser.add_argument('--legacy-optimizer',action='store_true')
    parser.add_argument('--disable-regional',action='store_true')
    parser.add_argument('--audit-dir')
    args=parser.parse_args()
    random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed);torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    data=ImageToImage2D(config.train_dataset,config.task_name,
        read_text(os.path.join(config.task_dataset,'Train_Val_text.xlsx')),RandomGenerator([224,224]),image_size=224)
    batch,names=next(iter(DataLoader(data,batch_size=16,num_workers=0)))
    torch.manual_seed(config.seed);torch.cuda.manual_seed_all(config.seed)
    model=BetterLViT(config.get_CTranS_config(),use_lora=False,
        text_encoder_name=config.text_encoder_name,text_seq_len=config.text_max_len).cuda().train()
    initial=digest(model.state_dict().items())
    rng=dict(cpu=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        cuda=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest())
    if args.calibrate:
        model.eval();records=measure(model)
        ratios=[s['raw_logit_gradient_ratio'] for r in records for s in r['gradients'].values()]
        assert all(math.isfinite(v) and v>0 for v in ratios)
        weight=math.floor(min(.25,.10/max(ratios))*1e6)/1e6
        assert weight>0 and initial==digest(model.state_dict().items())
        print(json.dumps(dict(status='ok',source_git_commit=config.source_git_commit,scope='Train only; no optimizer updates',
            initial_base_sha256=initial,common_weight=weight,max_raw_ratio=max(ratios),
            max_weighted_ratio=weight*max(ratios),formal_training_performed=False,test_split_accessed=False,batches=records)))
        return
    batch={k:v.cuda() for k,v in batch.items()}
    if args.legacy_optimizer:
        optimizer=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=3e-4,weight_decay=1e-4)
    else:
        groups,_,_=build_optimizer_parameter_groups(model,config.weight_decay)
        optimizer=torch.optim.Adam(groups,lr=config.learning_rate)
    objective=WeightedDiceFocal() if config.regional_mode=='none' else RegionalOverlapObjective(
        config.regional_mode,0 if args.disable_regional else config.regional_weight)
    losses=[];outputs=[];seconds=[]
    for step in range(5):
        torch.cuda.synchronize();started=time.time();optimizer.zero_grad()
        p=model(batch['image'],batch['input_ids'],batch['attention_mask'])
        loss=objective(p,batch['label'].float());loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
        optimizer.step();torch.cuda.synchronize()
        seconds.append(time.time()-started);losses.append(loss.item())
        outputs.append(hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest())
        del p,loss
        if args.audit_dir and step==1:run_audit(model,objective,0,args.audit_dir)
    print(json.dumps(dict(status='ok',source_git_commit=config.source_git_commit,profile=config.experiment_name,
        weight=config.regional_weight,seed=config.seed,formal_training_performed=False,test_split_accessed=False,
        initial_base_sha256=initial,initial_rng=rng,input_image_sha256=hashlib.sha256(batch['image'].cpu().numpy().tobytes()).hexdigest(),
        output_sha256_each_step=outputs,first_output_sha256=outputs[0],loss_each_step=losses,
        seconds_each_step=seconds,steady_seconds_per_batch=sum(seconds[2:])/3,
        temporary_optimizer_steps=5,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        audit_exercised=bool(args.audit_dir),regional_disabled=args.disable_regional,legacy_optimizer=args.legacy_optimizer)))


if __name__=='__main__':main()

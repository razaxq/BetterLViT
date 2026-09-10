"""Five temporary steps, formal optimizer groups, optional state-preserving audit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
import Config as config
from Load_Dataset import ImageToImage2D,RandomGenerator
from nets.BetterLViT import BetterLViT
from train_model import build_optimizer_parameter_groups
from utils import read_text,WeightedDiceFocal
from training_recipe import recipe_metadata
from tools.visual_aux_audit import digest,run_audit
from visual_aux_objective import VisualAuxiliaryObjective


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit-dir')
    parser.add_argument('--disable-aux-loss',action='store_true')
    args=parser.parse_args()
    random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed);torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    dataset=ImageToImage2D(config.train_dataset,config.task_name,
        read_text(os.path.join(config.task_dataset,'Train_Val_text.xlsx')),RandomGenerator([224,224]),image_size=224)
    batch,names=next(iter(DataLoader(dataset,batch_size=16,num_workers=0)))
    torch.manual_seed(config.seed);torch.cuda.manual_seed_all(config.seed)
    model=BetterLViT(config.get_CTranS_config(),use_lora=False,
        text_encoder_name=config.text_encoder_name,text_seq_len=config.text_max_len).cuda().train()
    common=digest((n,v) for n,v in model.state_dict().items() if not n.startswith('race.'))
    aux=digest((n,v) for n,v in model.state_dict().items() if n.startswith('race.'))
    rng=dict(cpu=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        cuda=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest())
    batch={k:v.cuda() for k,v in batch.items()}
    groups,decay,no_decay=build_optimizer_parameter_groups(model,config.weight_decay)
    optimizer=torch.optim.Adam(groups,lr=config.learning_rate)
    visual=config.visual_aux_mode!='none'
    objective=VisualAuxiliaryObjective(config.visual_aux_mode) if visual else WeightedDiceFocal()
    losses=[];outputs=[];durations=[];main_losses=[];gradient_checks=[]
    for step in range(5):
        torch.cuda.synchronize();started=time.time();optimizer.zero_grad()
        out=model(batch['image'],batch['input_ids'],batch['attention_mask'],return_aux=visual,
            race_slot_targets=batch['race_slot_targets'],race_zone_basis=batch['race_zone_basis'])
        pred=out['final'] if visual else out
        loss=objective.segmentation(pred,batch['label'].float()) if args.disable_aux_loss and visual else objective(out,batch['label'].float())
        main=objective.segmentation(pred,batch['label'].float()) if visual else loss
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if visual:
            assert all(p.grad is None for p in model.race.slot_head.parameters())
            if not args.disable_aux_loss:
                assert model.race.routes[0].evidence[0].weight.grad.abs().max()>0
                if config.visual_aux_mode=='regional': assert model.race.routes[0].presence.weight.grad.abs().max()>0
                else: assert model.race.routes[0].presence.weight.grad is None
        optimizer.step();torch.cuda.synchronize()
        durations.append(time.time()-started);losses.append(loss.item());main_losses.append(main.item())
        outputs.append(hashlib.sha256(pred.detach().cpu().numpy().tobytes()).hexdigest())
        del out,pred,loss,main
        if args.audit_dir and step==1:run_audit(model,objective,0,args.audit_dir)
    print(json.dumps(dict(status='ok',profile=config.experiment_name,source_git_commit=config.source_git_commit,
        seed=config.seed,training_recipe=recipe_metadata(config),formal_training_performed=False,
        test_split_accessed=False,temporary_optimizer_steps=5,batch_size=16,
        initial_common_sha256=common,initial_aux_sha256=aux,initial_rng=rng,
        input_image_sha256=hashlib.sha256(batch['image'].cpu().numpy().tobytes()).hexdigest(),
        first_output_sha256=outputs[0],output_sha256_each_step=outputs,loss_each_step=losses,main_loss_each_step=main_losses,
        seconds_each_step=durations,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        audit_exercised=bool(args.audit_dir),aux_loss_disabled=args.disable_aux_loss,
        optimizer_groups=dict(decay=len(decay),no_decay=len(no_decay)))))


if __name__=='__main__':main()

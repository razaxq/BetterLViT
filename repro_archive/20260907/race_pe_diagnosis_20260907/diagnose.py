"""Read-only RACE-PE attribution on Train/Val; no Test data or optimizer."""
import os
os.environ.update(BETTERLVIT_EXPERIMENT='p9_race_pe', TEST_SPLIT_ALLOWED='0',
    HF_HOME='/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface',
    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
import sys, json, hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path('/root/BetterLViT-race-pe-p9')
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
import Config as config
from export_validation_metrics import build_model, validation_loader
from race_pe_objective import RACEPEObjective
from race_semantics import parse_report_slots_pe
from utils import read_text

OUT = Path(__file__).resolve().parent
SOURCE = Path('/root/race_pe_runs/c4_p9_20260906')


def metrics(prob, label):
    pred = prob > .5
    dims = (1, 2)
    tp = (pred & label).sum(dims).double()
    pp, gt = pred.sum(dims).double(), label.sum(dims).double()
    union = pp + gt - tp
    return dict(dice=2*tp/(pp+gt).clamp_min(1), iou=tp/union.clamp_min(1),
        precision=tp/pp.clamp_min(1), recall=tp/gt.clamp_min(1),
        fp=pp-tp, fn=gt-tp, prediction_pixels=pp, label_pixels=gt,
        brier=(prob-label.float()).square().mean(dims).double())


def main():
    torch.manual_seed(1219)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    reference = json.loads((SOURCE/'p9_validation.json').read_text())
    ckpt = torch.load(reference['checkpoint'], map_location='cpu', weights_only=True)
    assert ckpt['source_git_commit'] == '8129c1f039ed77f78e70305aca0bb9708b3b56b1'
    model = build_model()
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.cuda().eval()
    history = [{k:v for k,v in row.items() if k in (
        'epoch','val_iou','val_dice','race_stats','train_loss_components','val_loss_components')}
        for row in ckpt['epoch_history']]
    del ckpt
    (OUT/'history.json').write_text(json.dumps(history, indent=2))
    train_text = read_text(Path(config.task_dataset)/'Train_Val_text.xlsx')
    train_names = sorted((ROOT/config.train_dataset/'labelcol').iterdir())
    labels = torch.stack([parse_report_slots_pe(train_text[f.name]) for f in train_names])
    label_audit = dict(samples=len(train_names),positive=int((labels[:,:6]==1).sum()),
        negative=int((labels[:,:6]==0).sum()),unknown=int((labels[:,:6]<0).sum()),
        no_known_location=int((labels[:,:6]<0).all(1).sum()))
    dataset, loader = validation_loader(16)
    modes = ['native','routes_off','all_one_slots','shuffled_slots']
    mode_records = {}
    slots_all, target_all, stats = [], [], [[] for _ in range(4)]
    native_prob = []
    original_strengths = [r.strength_logit.detach().clone() for r in model.race.routes]
    for mode in modes:
        model.race.route_enabled = mode != 'routes_off'
        def hook(module, inputs, output):
            if mode == 'all_one_slots':
                output = output.clone(); output[:,:6] = 30
            elif mode == 'shuffled_slots':
                output = output.clone(); output[:,:6] = output[:,:6].roll(1,0)
            return output
        handle = model.race.slot_head.register_forward_hook(hook)
        records=[]; absdiff_sum=0.; flips=0; npixels=0
        with torch.inference_mode():
            for bi,(batch,names) in enumerate(loader):
                b={k:v.cuda() for k,v in batch.items()}
                aux=model(b['image'],b['input_ids'],b['attention_mask'],return_aux=True,
                    race_slot_targets=b['race_slot_targets'],race_zone_basis=b['race_zone_basis'])
                prob=aux['final'][:,0].cpu(); mask=batch['label'].bool()
                met=metrics(prob,mask)
                for i,name in enumerate(names):
                    records.append(dict(name=name,**{k:float(v[i]) for k,v in met.items()}))
                if mode == 'native':
                    native_prob.append(prob.clone())
                    slots_all.append(aux['slot_logits'][:,:6].sigmoid().cpu())
                    target_all.append(batch['race_slot_targets'][:,:6])
                    for si,r in enumerate(aux['pe_routes']):
                        target=F.adaptive_avg_pool2d(b['label'].float().unsqueeze(1),r['extent_logits'].shape[-2:])
                        basis=r['basis']; mass=basis.sum((2,3)); occ=(target*basis).sum((2,3))/mass.clamp_min(1)
                        pres=r['presence_logits'].sigmoid(); extent=r['extent_logits'].sigmoid()
                        valid=mass>0; positive=(occ>0)&valid; negative=(occ==0)&valid
                        known=(b['race_slot_targets'][:,:6]==1)&valid
                        stats[si].append(dict(batch=len(names),
                            region_count=int(valid.sum()),true_present=int(positive.sum()),
                            predicted_present=int(((pres>.5)&valid).sum()),
                            presence_fp=int(((pres>.5)&negative).sum()),
                            presence_tn=int(negative.sum()),text_positive=int(known.sum()),
                            text_mask_conflicts=int((known&negative).sum()),
                            occupancy_mae=float(((r['occupancy']-occ).abs()*valid).sum()),
                            extent_bg_sum=float((extent*(1-target)).sum()),bg_mass=float((1-target).sum()),
                            extent_fg_sum=float((extent*target).sum()),fg_mass=float(target.sum())))
                else:
                    native=native_prob[bi]
                    absdiff_sum+=float((prob-native).abs().double().sum())
                    flips+=int(((prob>.5)!=(native>.5)).sum());npixels+=prob.numel()
                if bi%30==0:print(mode,bi,len(loader),flush=True)
        handle.remove()
        assert len(records)==1429
        mode_records[mode]=records
        mean={k:float(np.mean([r[k] for r in records])) for k in records[0] if k!='name'}
        if mode!='native': mean.update(mean_abs_probability_change=absdiff_sum/npixels,mask_flip_fraction=flips/npixels)
        (OUT/(mode+'.json')).write_text(json.dumps(dict(split='validation',test_split_accessed=False,
            checkpoint_git_commit=reference['checkpoint_git_commit'],mode=mode,mean=mean,records=records),indent=2))
        print(mode,mean,flush=True)
    assert abs(np.mean([r['iou'] for r in mode_records['native']])-reference['macro_iou'])<1e-10
    z=torch.cat(slots_all); targets=torch.cat(target_all)
    slot_summary=dict(mean=float(z.mean()),std=float(z.std()),minimum=float(z.min()),
        fraction_above_099=float((z>.99).float().mean()),per_zone_mean=z.mean(0).tolist(),
        per_zone_std=z.std(0).tolist(),known_mean=float(z[targets==1].mean()),unknown_mean=float(z[targets<0].mean()))
    region_summary=[]
    for rows in stats:
        sums={k:sum(r[k] for r in rows) for k in rows[0]}
        sums.update(occupancy_mae=sums['occupancy_mae']/sums['region_count'],
            extent_background_mean=sums['extent_bg_sum']/sums['bg_mass'],
            extent_foreground_mean=sums['extent_fg_sum']/sums['fg_mass'])
        region_summary.append(sums)
    # A limited gradient probe is descriptive, not a causal training ablation.
    model.race.route_enabled=True
    objective=RACEPEObjective(); gradients=[]
    param=next(p for n,p in model.named_parameters() if n.startswith('inc.') and p.ndim==4)
    for bi,(batch,names) in enumerate(loader):
        if bi==4:break
        b={k:v[:4].cuda() for k,v in batch.items()}
        aux=model(b['image'],b['input_ids'],b['attention_mask'],return_aux=True,
            race_slot_targets=b['race_slot_targets'],race_zone_basis=b['race_zone_basis'])
        total=objective(aux,b['label']);main_loss=objective.segmentation(aux['final'],b['label'])
        gm=torch.autograd.grad(main_loss,param,retain_graph=True)[0].flatten()
        ga=torch.autograd.grad(total-main_loss,param)[0].flatten()
        gradients.append(dict(names=list(names[:4]),cosine=float(F.cosine_similarity(gm,ga,dim=0)),
            auxiliary_to_main_norm=float(ga.norm()/gm.norm().clamp_min(1e-12))))
    result=dict(split='train_and_validation',test_split_accessed=False,training_performed=False,
        source_git_commit=reference['checkpoint_git_commit'],checkpoint_best_epoch=reference['checkpoint_best_epoch'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        training_label_audit=label_audit,validation_slots=slot_summary,visual_heads=region_summary,
        gradient_probe=gradients,permutation='cyclic roll within validation batch; only RACE slot head output modified')
    (OUT/'diagnosis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':
    try:
        main();(OUT/'status').write_text('complete')
    except BaseException:
        (OUT/'status').write_text('failed');raise

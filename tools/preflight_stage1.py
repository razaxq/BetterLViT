"""Real Train-only forward/backward and common-initialization verification."""
import argparse,copy,gc,hashlib,json,os,random,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--reference-repo',type=Path);p.add_argument('--initialization-only',action='store_true')
args=p.parse_args();repo=args.reference_repo or Path(__file__).resolve().parents[1]
sys.path.insert(0,str(repo));os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch.utils.data import DataLoader
import Config as config
from nets.LViT import LViT
from nets.BetterLViT import BetterLViT
from Load_Dataset import ImageToImage2D,RandomGenerator
from utils import read_text,WeightedDiceFocal,RACEObjective

def fingerprint(state):
    h=hashlib.sha256()
    for name,t in sorted(state.items()):h.update(name.encode());h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def main():
    torch.set_num_threads(1)
    random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    cfg=config.get_CTranS_config()
    initial=LViT(cfg,text_seq_len=32)
    initial_fingerprint=fingerprint(initial.state_dict())
    shared_count=None
    if getattr(cfg,'stage1_match_fsdr_initialization',False):
        ref_cfg=copy.deepcopy(cfg);ref_cfg.decoder_fusion_mode='fam_eppa_v4b';ref_cfg.stage1_match_fsdr_initialization=False
        torch.manual_seed(config.seed);ref=LViT(ref_cfg,text_seq_len=32)
        actual=initial.state_dict();expected=ref.state_dict()
        common=actual.keys() & expected.keys();assert common
        assert all(actual[k].shape==expected[k].shape and torch.equal(actual[k],expected[k]) for k in common)
        assert all('.pixModule.' in k for k in actual.keys()-expected.keys())
        assert all('.eppa.' in k for k in expected.keys()-actual.keys())
        shared_count=len(common)
        assert all(hasattr(getattr(initial,s),'pixModule') and not hasattr(getattr(initial,s),'eppa') for s in ('up4','up3','up2','up1'))
        del ref,actual,expected
    del initial;gc.collect()
    if args.initialization_only:
        print(json.dumps(dict(experiment=config.experiment_name,seed=config.seed,initialization_sha256=initial_fingerprint,shared_tensors_verified=shared_count)));return
    random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed);torch.cuda.manual_seed_all(config.seed)
    dataset=ImageToImage2D(config.train_dataset,config.task_name,
        read_text(os.path.join(config.task_dataset,'Train_Val_text.xlsx')),
        RandomGenerator([224,224]),image_size=224)
    batch,names=next(iter(DataLoader(dataset,batch_size=16,num_workers=0)))
    torch.manual_seed(config.seed);torch.cuda.manual_seed_all(config.seed)
    model=BetterLViT(cfg,n_channels=3,n_classes=1,text_encoder_name=config.text_encoder_name,text_seq_len=32,use_lora=False).cuda().train()
    assert all(not p.requires_grad for p in model.text_encoder.parameters())
    b={k:v.cuda() for k,v in batch.items()}
    pred=model(b['image'],b['input_ids'],b['attention_mask'],return_aux=config.race_enabled,
        race_slot_targets=b['race_slot_targets'],race_zone_basis=b['race_zone_basis'])
    objective=RACEObjective(aux_weight=.05) if config.race_enabled else WeightedDiceFocal()
    loss=objective(pred,b['label']);loss.backward(retain_graph=config.race_enabled)
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().max()>0 for p in model.down1.parameters())
    if config.race_enabled:
        assert getattr(model.race,'route_enabled',True)
        assert all(r.strength_logit.grad is not None and r.strength_logit.grad.abs().max()>0 for r in model.race.routes)
        assert model.race.slot_head[0].weight.grad.abs().max()>0
        model.zero_grad(set_to_none=True)
        objective(dict(pred,final=pred['final'].detach()),b['label']).backward()
        assert any(p.grad is not None and p.grad.abs().max()>0 for p in model.down1.parameters())
        assert all(r.evidence[-1].weight.grad.abs().max()>0 for r in model.race.routes)
    else:assert model.race is None and not isinstance(pred,dict)
    final=pred['final'] if isinstance(pred,dict) else pred
    assert final.shape==(16,1,224,224) and final.shape[-2:]==b['label'].shape[-2:]
    print(json.dumps(dict(status='ok',experiment=config.experiment_name,seed=config.seed,batch_size=16,
        test_split_accessed=False,training_performed=False,loss=loss.item(),
        output_sha256=hashlib.sha256(final.detach().cpu().numpy().tobytes()).hexdigest(),
        initialization_sha256=initial_fingerprint,shared_tensors_verified=shared_count,
        race_route_enabled=bool(config.race_enabled),race_auxiliary_enabled=bool(config.race_enabled),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)))

if __name__=='__main__':main()

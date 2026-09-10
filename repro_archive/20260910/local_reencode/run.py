"""Frozen R2, matched feature/image re-encoding feasibility experiment."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE/'manifest.json').read_text())
ROOT = Path(MANIFEST['baseline_repository'])
os.environ.update(BETTERLVIT_EXPERIMENT='r2_single_cosine', TEST_SPLIT_ALLOWED='0', AUTO_TEST_EVALUATE='0',
                  HF_HOME='/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface',
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                  PYTHONHASHSEED='1219', TOKENIZERS_PARALLELISM='false')


def guard(event, args):
    if event in ('open','os.listdir','os.scandir') and args and isinstance(args[0], (str,bytes,os.PathLike)):
        parts = Path(os.fsdecode(args[0])).parts
        if 'Test_Folder' in parts or 'Test_text.xlsx' in parts:
            raise RuntimeError('Test access prohibited in this feasibility experiment')


sys.addaudithook(guard)
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
os.chdir(ROOT)
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
import Config as config
from Load_Dataset import ImageToImage2D, ValGenerator
from export_validation_metrics import build_model, validation_loader
from utils import read_text, WeightedDiceFocal
sys.path.insert(0, str(HERE))
from model import Reencoder, region_mask
from analysis import digest, metrics, summarize, write_json


def deterministic():
    torch.manual_seed(1219); np.random.seed(1219)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def state_hash(model):
    h = hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Features:
    def __init__(self, model):
        self.model, self.values = model, {}
        self.handles = []
        for name,module in [('decoder',model.up1),('fine',model.down1),('semantic',model.reconstruct2),('logit',model.outc)]:
            def hook(mod, inp, out, key=name):
                self.values[key] = F.avg_pool2d(out,2).detach() if key=='decoder' else out.detach()
            self.handles.append(module.register_forward_hook(hook))

    def batch(self, batch):
        self.values.clear()
        with torch.inference_mode():
            p = self.model(batch['image'].cuda(),batch['input_ids'].cuda(),batch['attention_mask'].cuda())
            feature = torch.cat([self.values[k] for k in ('decoder','fine','semantic')],1)
            assert feature.shape[1:] == (320,112,112)
            assert torch.equal(p,self.values['logit'].sigmoid())
            return dict(feature=feature.cpu().half(), logit=self.values['logit'].cpu(),
                        image=batch['image'].cpu().half(), label=batch['label'].cpu().byte())


def collect(extractor, loader):
    cache, names, count = {}, [], 0
    for batch,ns in loader:
        values = extractor.batch(batch)
        if not cache:
            cache = {k:torch.empty((len(loader.dataset),)+v.shape[1:],dtype=v.dtype) for k,v in values.items()}
        for k,v in values.items(): cache[k][count:count+len(ns)].copy_(v)
        names.extend(ns); count += len(ns)
    assert count == len(loader.dataset)
    return cache,names


def normalization(x):
    sums = torch.zeros(320,dtype=torch.float64); squares = sums.clone(); count = 0
    for batch in x.split(16):
        batch = batch.float()
        variance,mean = torch.var_mean(batch,dim=(0,2,3),unbiased=False)
        n = batch.shape[0]*batch.shape[2]*batch.shape[3]
        sums += mean.double()*n; squares += (variance.double()+mean.double().square())*n; count += n
    mean = sums/count; std = (squares/count-mean.square()).clamp_min(0).sqrt().clamp_min(1e-3)
    return mean.float()[None,:,None,None],std.float()[None,:,None,None]


def gpu_batch(cache, indices):
    return {k:v[indices].float().cuda() for k,v in cache.items()}


def audit_loss(heads, cache, objective):
    values = {k:[] for k in heads}; values['baseline']=[]
    with torch.no_grad():
        for start in range(0,min(64,len(cache['label'])),16):
            batch = gpu_batch(cache,np.arange(start,min(start+16,len(cache['label']))))
            values['baseline'].append(float(objective(batch['logit'].sigmoid(),batch['label'])))
            for arm,head in heads.items(): values[arm].append(float(objective(head(batch,arm)['probability'],batch['label'])))
    return {k:float(np.mean(v)) for k,v in values.items()}


def fit(cache, steps, progress=None):
    mean,std = normalization(cache['feature'])
    torch.manual_seed(1219)
    original = Reencoder(mean,std)
    heads = {arm:copy.deepcopy(original).cuda().train() for arm in ('features','image')}
    initial_hash = state_hash(original)
    assert all(state_hash(h)==initial_hash for h in heads.values())
    optimizers = {k:torch.optim.Adam(h.parameters(),lr=3e-4,weight_decay=1e-4) for k,h in heads.items()}
    objective = WeightedDiceFocal(dice_weight=.5,focal_weight=.5)
    rng = np.random.default_rng(1219)
    batch = gpu_batch(cache,np.arange(min(16,len(cache['label']))))
    for arm,head in heads.items():
        out = head(batch,arm)
        assert torch.equal(out['probability'],batch['logit'].sigmoid())
        assert torch.all(out['mask'].sum((1,2,3))==40*16*16)
    before = audit_loss(heads,cache,objective)
    history = []; grad_modules = {}
    torch.cuda.synchronize(); started = time.time()
    for step in range(steps):
        indices = rng.integers(0,len(cache['label']),size=16)
        batch = gpu_batch(cache,indices)
        lr = 1e-6+.5*(3e-4-1e-6)*(1+math.cos(math.pi*step/max(1,steps-1)))
        row = dict(step=step+1,lr=lr)
        for arm,head in heads.items():
            opt = optimizers[arm]
            for g in opt.param_groups: g['lr']=lr
            opt.zero_grad(set_to_none=True)
            out = head(batch,arm)
            loss = objective(out['probability'],batch['label'])
            assert torch.isfinite(loss)
            loss.backward()
            if step==2:
                grad_modules[arm] = {name:any(p.grad is not None and bool(p.grad.abs().max()>0) for p in mod.parameters())
                                     for name,mod in head.named_children()}
                assert all(grad_modules[arm].values()), grad_modules[arm]
            opt.step(); row[arm]=float(loss.detach())
        history.append(row)
        if progress is not None and ((step+1)%100==0 or step==steps-1):
            elapsed = time.time()-started
            progress.update(phase='training',step=step+1,total_steps=steps,training_seconds=elapsed,
                            remaining_training_seconds=elapsed/(step+1)*(steps-step-1))
            write_json(ARGS.output/'progress.json',progress)
            print(json.dumps(row),flush=True)
    torch.cuda.synchronize(); seconds_per_step=(time.time()-started)/steps
    for h in heads.values(): h.eval()
    after = audit_loss(heads,cache,objective)
    return heads,dict(history=history,seconds_per_paired_step=seconds_per_step,initial_shared_hash=initial_hash,
                      parameter_count=sum(p.numel() for p in original.parameters()),fit_64_loss_before=before,
                      fit_64_loss_after=after,modules_receive_gradients=grad_modules)


def evaluate(extractor, heads, loader):
    records=[]
    for batch,names in loader:
        cache=extractor.batch(batch)
        data=gpu_batch(cache,np.arange(len(names)))
        with torch.no_grad():
            base=data['logit'].sigmoid()[:,0].cpu().numpy()
            outputs={arm:head(data,arm) for arm,head in heads.items()}
            masks=outputs['image']['mask'][:,0].cpu().numpy().astype(bool)
            p={arm:out['probability'][:,0].cpu().numpy() for arm,out in outputs.items()}
            full={arm:out['full_probability'][:,0].cpu().numpy() for arm,out in outputs.items()}
            for arm in heads:
                assert np.array_equal(p[arm][~masks],base[~masks]),'Outside selected regions changed'
                assert np.all(masks.sum((1,2))==10240)
            for i,name in enumerate(names):
                label=batch['label'][i].numpy().astype(bool)
                error=(base[i]>.5)!=label
                # Old bounded residual can change class only when |base logit| <= 0.5.
                old_band=np.abs(cache['logit'][i,0].numpy())<=.5
                row=dict(name=name,baseline=metrics(base[i],label),
                         **{arm:metrics(p[arm][i],label) for arm in heads},
                         full_diagnostic={arm:metrics(full[arm][i],label) for arm in heads},
                         selected_pixels=int(masks[i].sum()),error_pixels=int(error.sum()),
                         selected_errors=int((error&masks[i]).sum()),
                         errors_outside_old_logit_bound=int((error&~old_band).sum()))
                records.append(row)
    return records


def main(args):
    started=time.time(); deterministic()
    assert not MANIFEST['test_split_allowed'] and not config.text_use_lora and config.boundary_loss_weight==0
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==MANIFEST['baseline_source_git_commit']
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()
    assert digest(HERE/'r2_validation.json')==MANIFEST['validation_sha256']
    ckpt=torch.load(MANIFEST['baseline_checkpoint'],map_location='cpu',weights_only=True)
    assert ckpt['source_git_commit']==MANIFEST['baseline_source_git_commit']
    model=build_model(); model.load_state_dict(ckpt['state_dict'],strict=True); del ckpt
    model.requires_grad_(False); model.cuda().eval()
    original_hash=state_hash(model); extractor=Features(model)
    text=read_text(os.path.join(config.task_dataset,'Train_Val_text.xlsx'))
    train=ImageToImage2D(config.train_dataset,config.task_name,text,ValGenerator([224,224]),image_size=224)
    assert len(train)==5716 and Path(train.dataset_path).resolve().name=='Train_Folder'
    indices=sorted(range(len(train)),key=lambda i:hashlib.sha256(('local-reencode-v1:'+train.mask_list[i]).encode()).hexdigest())
    fit_ix,audit_ix=indices[:-256],indices[-256:]
    assert len(fit_ix)==5460 and not set(fit_ix)&set(audit_ix)
    loader=lambda ix:DataLoader(Subset(train,ix),batch_size=16,shuffle=False,num_workers=0)
    feature_started=time.time()
    cache,names=collect(extractor,loader(fit_ix[:16] if args.preflight else fit_ix))
    torch.cuda.synchronize(); feature_per_image=(time.time()-feature_started)/len(names)
    if args.preflight:
        heads,fit1=fit(cache,8)
        hashes={k:state_hash(v) for k,v in heads.items()}; del heads
        heads,fit2=fit(cache,8)
        assert {k:state_hash(v) for k,v in heads.items()}==hashes
        assert fit1['history']==fit2['history']
        test_mask=region_mask(torch.full((1,1,224,224),.5,device='cuda'))
        assert torch.equal(test_mask[0,0,:32],torch.ones_like(test_mask[0,0,:32]))
        assert test_mask.sum()==10240
        assert state_hash(model)==original_hash and all(p.grad is None for p in model.parameters())
        eval_start=time.time(); evaluate(extractor,heads,loader(fit_ix[:16])); torch.cuda.synchronize()
        eval_seconds=(time.time()-eval_start)/16
        predicted=120+1.2*(feature_per_image*5460+fit1['seconds_per_paired_step']*MANIFEST['steps']+eval_seconds*(256+1429))
        result=dict(status='ok',source_git_commit=args.source_sha,deterministic_two_repeats=True,
                    initialization_identity=True,shared_initialization=True,frozen_state_unchanged=True,
                    mask_budget_verified=True,outside_mask_invariant=True,fit=fit1,head_hashes=hashes,
                    feature_seconds_per_image=feature_per_image,evaluation_seconds_per_image=eval_seconds,
                    predicted_run_seconds=predicted,gpu_peak_bytes=torch.cuda.max_memory_allocated(),
                    test_split_accessed=False,scripts_sha256={p.name:digest(p) for p in HERE.glob('*.py')})
        write_json(args.output,result);print(json.dumps(result));return
    heads,training=fit(cache,MANIFEST['steps'],dict(source_git_commit=args.source_sha))
    training.update(fit_names=names,audit_names=[train.mask_list[i] for i in audit_ix],
                    cache_bytes=sum(v.numel()*v.element_size() for v in cache.values()),
                    feature_seconds_per_image=feature_per_image)
    del cache
    write_json(args.output/'training.json',training)
    write_json(args.output/'audit_records.json',dict(split='train_branch_audit',records=evaluate(extractor,heads,loader(audit_ix))))
    dataset,val_loader=validation_loader(16)
    assert len(dataset)==1429 and Path(dataset.dataset_path).resolve().name=='Val_Folder'
    records=evaluate(extractor,heads,val_loader)
    refs={r['name']:r for r in json.loads((HERE/'r2_validation.json').read_text())['records']}
    assert set(refs)=={r['name'] for r in records} and not set(names)&set(refs)
    maximum=max(abs(r['baseline'][k]-refs[r['name']][k]) for r in records for k in ('iou','dice','precision','recall'))
    assert maximum<1e-12,maximum
    assert state_hash(model)==original_hash and all(p.grad is None for p in model.parameters())
    metadata=dict(source_git_commit=args.source_sha,baseline_source_git_commit=MANIFEST['baseline_source_git_commit'],
                  baseline_checkpoint_sha256=digest(MANIFEST['baseline_checkpoint']),baseline_state_sha256=original_hash,
                  baseline_state_unchanged=True,baseline_per_image_max_difference=maximum,
                  split='validation',test_split_accessed=False)
    write_json(args.output/'records.json',dict(**metadata,records=records))
    write_json(args.output/'summary.json',summarize(records))
    for arm,head in heads.items():torch.save(dict(state_dict=head.cpu().state_dict(),metadata=metadata,arm=arm),args.output/(arm+'.pt'))
    print(json.dumps(dict(phase='complete',elapsed_seconds=time.time()-started,baseline_max_difference=maximum)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--source-sha',required=True)
    ARGS=parser.parse_args()
    assert len(ARGS.source_sha)==40 and all(c in '0123456789abcdef' for c in ARGS.source_sha)
    if ARGS.preflight:main(ARGS)
    else:
        ARGS.output.mkdir(exist_ok=False)
        runtime=dict(phase='running',started_unix=time.time(),source_git_commit=ARGS.source_sha,test_split_accessed=False)
        write_json(ARGS.output/'runtime.json',runtime)
        try:
            main(ARGS)
            runtime.update(phase='complete',completed_unix=time.time(),
                           artifacts={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in ARGS.output.iterdir() if p.is_file() and p.name!='runtime.json'})
        except BaseException:
            runtime.update(phase='failed',completed_unix=time.time(),traceback=traceback.format_exc());raise
        finally:write_json(ARGS.output/'runtime.json',runtime)

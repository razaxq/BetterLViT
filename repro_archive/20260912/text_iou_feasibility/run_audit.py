"""No-update E0/E1 audit of immutable Train caches and completed AdamW heads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONHASHSEED='1219')
HERE = Path(__file__).resolve().parent
M = json.loads((HERE/'manifest.json').read_text())
B = Path(M['b_directory']); D = Path(M['d_directory'])

def guard(event, args):
    if event in ('open','os.listdir','os.scandir') and args and isinstance(args[0],(str,bytes,os.PathLike)):
        parts = [p.lower() for p in Path(os.fsdecode(args[0])).parts]
        if any(p in parts for p in ('val_folder','test_folder','val_text.xlsx','test_text.xlsx')):
            raise RuntimeError('Val/Test prohibited in E0/E1')
sys.addaudithook(guard)
sys.path[:0] = [str(HERE), '/root/BetterLViT-recipe-r2']
import numpy as np
import torch
from heads import ResidualHead, VARIANTS, prediction, interpolation_matrix
from analysis import digest, metrics, write_json
from utils import WeightedDiceFocal

def binary_metrics(pred, y):
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
    return dict(tp=tp,fp=fp,fn=fn,iou=tp/(tp+fp+fn) if tp+fp+fn else 0.)

def gradient_stats(g, y, z, base):
    # FP should decrease: positive gradient; FN should increase: negative gradient.
    desired = torch.where(y.bool(), -1., 1.)
    categories = dict(fp=base & ~y.bool(), fn=~base & y.bool(), tp=base & y.bool(), tn=~base & ~y.bool())
    signed = g * desired
    result = {}
    for name, mask in categories.items():
        for window, selected in (('all',mask),('near',mask & (z.abs()<=.5))):
            result[name+'/'+window] = dict(n=int(selected.sum()), wrong=int(((signed<0)&selected).sum()),
                zero=int(((signed==0)&selected).sum()), abs_gradient=float(g[selected].abs().double().sum()),
                wrong_abs_gradient=float(g[(signed<0)&selected].abs().double().sum()))
    return result

def main(args):
    torch.set_num_threads(4); torch.manual_seed(1219); np.random.seed(1219)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    for root,key in ((B,'b_hashes'),(D,'d_hashes')):
        for name,expected in M[key].items(): assert digest(root/name)==expected, name
    assert digest(HERE/'selection.json')==M['selection_sha256']
    assert digest(HERE/'heads.py')==M['heads_sha256']
    repo = Path('/root/BetterLViT-recipe-r2')
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==M['baseline_source_git_commit']
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
    rows = json.loads((B/'split.json').read_text())['records']
    selection = json.loads((HERE/'selection.json').read_text())
    eligible = selection['all_eligible_fit']; audit = selection['original32']+selection['additional128']
    assert len(eligible)==1930 and len(set(audit))==160 and set(audit)<=set(eligible)
    assert all(rows[i]['eligible'] and rows[i]['partition']=='fit' for i in eligible)
    cache = json.loads((B/'results/cache_manifest.json').read_text())
    for name,meta in cache['files'].items(): assert digest(B/'cache'/name)==meta['sha256'],name
    arrays = {k:np.memmap(B/'cache'/(k+'.bin'),dtype=dtype,mode='r',shape=tuple(shape)) for k,(shape,dtype) in cache['shapes'].items()}
    text = np.load(B/'cache/text.npz'); embeddings=torch.from_numpy(text['embeddings']).cuda(); masks=torch.from_numpy(text['masks']).cuda()
    lookup={text:i for i,text in enumerate(cache['unique_texts'])}
    def data(ix,features=False):
        assert set(ix)<=set(eligible), 'Non-fit cache index blocked'
        z=torch.from_numpy(np.array(arrays['logits'][ix])).cuda()
        labels=np.unpackbits(np.array(arrays['masks'][ix]),axis=1,count=224*224,bitorder='little').reshape(-1,1,224,224)
        y=torch.from_numpy(labels).cuda().float()
        if not features:return z,y
        f=torch.from_numpy(np.array(arrays['features'][ix])).cuda().float()
        c=[lookup[rows[i]['canonical']] for i in ix]; r=[lookup[rows[i]['reference']] for i in ix]
        return (f,embeddings[c],masks[c],embeddings[r],masks[r],torch.ones(len(ix),device='cuda',dtype=torch.bool)),z,y
    e0=[]
    with torch.no_grad():
        for start in range(0,len(eligible),16):
            ix=eligible[start:start+16]; z,y=data(ix); base=z.sigmoid()>.5
            for j,i in enumerate(ix):
                truth=y[j].bool(); original=binary_metrics(base[j],truth)
                r=dict(index=i,name=rows[i]['name'],baseline=original,oracles={})
                for name,bound in (('unprojected',.5),('projected_loose',1.000001)):
                    best=torch.where(truth,(z[j]+bound).sigmoid()>.5,(z[j]-bound).sigmoid()>.5)
                    out=binary_metrics(best,truth)
                    assert out['iou']>=original['iou'] and out['fp']<=original['fp'] and out['fn']<=original['fn']
                    r['oracles'][name]=out
                e0.append(r)
    write_json(args.output/'e0_per_image.json',e0)
    e0_summary=dict(n=len(e0),baseline_iou=float(np.mean([r['baseline']['iou'] for r in e0])),oracles={})
    for name in ('unprojected','projected_loose'):
        gain=float(np.mean([r['oracles'][name]['iou']-r['baseline']['iou'] for r in e0]))
        e0_summary['oracles'][name]=dict(eligible_mean_gain=gain,all_fit_folded_gain=gain*1930/M['fit_count'],
            fp_correctable_fraction=sum(r['baseline']['fp']-r['oracles'][name]['fp'] for r in e0)/sum(r['baseline']['fp'] for r in e0),
            fn_correctable_fraction=sum(r['baseline']['fn']-r['oracles'][name]['fn'] for r in e0)/sum(r['baseline']['fn'] for r in e0))
    write_json(args.output/'e0_summary.json',e0_summary)
    print(json.dumps(dict(event='e0_complete',summary=e0_summary)),flush=True)
    assert e0_summary['oracles']['projected_loose']['all_fit_folded_gain']>=.003, 'STOP: amplitude upper bound too small'
    ckpt=torch.load(D/'results/heads.pt',map_location='cpu',weights_only=False)
    assert ckpt['source_git_commit']==M['d_source_git_commit'] and ckpt['steps']==512
    heads={}
    for v in VARIANTS:
        head=ResidualHead().cuda();head.load_state_dict(ckpt['heads']['adamw/'+v],strict=True); head.eval()
        head.requires_grad_(False);heads[v]=head
    original_d = json.loads((D/'results/train_diagnostics.json').read_text())[-1]['cases']
    criterion=WeightedDiceFocal(); W=interpolation_matrix().cuda();e1=[]; gradients=[]; matched=[]
    for start in range(0,len(audit),16):
        ix=audit[start:start+16]; inputs,z,y=data(ix,True); p=z.sigmoid(); base=p>.5
        batch=[]
        for j,i in enumerate(ix):
            batch.append(dict(index=i,name=rows[i]['name'],cohort='original32' if i in selection['original32'] else 'additional128',
                baseline=metrics(p[j].cpu().numpy(),y[j].cpu().numpy()),conditions={}))
        p_grad=p.detach().requires_grad_(True)
        losses={'dice':criterion.dice_loss(p_grad,y),'focal':criterion.focal_loss(p_grad,y),'total':criterion(p_grad,y)}
        for lossname,loss in losses.items():
            gp=torch.autograd.grad(loss,p_grad,retain_graph=True)[0]*len(ix)
            w=p*(1-p);g=gp*w
            mean=(gp*w).double().sum((1,2,3),keepdim=True)/w.double().sum((1,2,3),keepdim=True).clamp_min(1e-30)
            proj=w*(gp-mean.to(gp.dtype)); smooth=W@(W.T@proj@W)@W.T
            # Actual effective logit update after re-solving lambda; spatial
            # coefficient descent itself is -smooth, then mass projection centers it.
            smooth_mean=(smooth*w).double().sum((1,2,3),keepdim=True)/w.double().sum((1,2,3),keepdim=True).clamp_min(1e-30)
            effective=smooth-smooth_mean.to(smooth.dtype)
            unrestricted_smooth=W@(W.T@g@W)@W.T
            for direction,value in (('unrestricted',g),('mass_projected',proj),
                ('unrestricted_28grid',unrestricted_smooth),('mass_projected_28grid',effective)):
                for j,i in enumerate(ix):
                    gradients.append(dict(index=i,cohort=batch[j]['cohort'],loss=lossname,direction=direction,
                        stats=gradient_stats(value[j],y[j],z[j],base[j])))
        with torch.no_grad():
            for v,head in heads.items():
                for condition in (('correct','swapped') if v!='image' else ('correct',)):
                    inp=inputs if condition=='correct' else (inputs[0],inputs[3],inputs[4],inputs[1],inputs[2],inputs[5])
                    delta=head(*inp,v);out=prediction(z,delta,v);err=base!=y.bool()
                    k=int(224*224*M['candidate_top_fraction'])
                    score=(out-p).abs().flatten(1)
                    ranked=torch.argsort(score,dim=1,descending=True,stable=True)[:,:k]
                    uncertain=torch.argsort((p*(1-p)).flatten(1),dim=1,descending=True,stable=True)[:,:k]
                    key=v+'/'+condition
                    matched.append(dict(indices=ix,cohort=batch[0]['cohort'],condition=key,
                        baseline_loss=float(criterion(p,y)),output_loss=float(criterion(out,y))))
                    for j,i in enumerate(ix):
                        b=base[j]; q=out[j]>.5; truth=y[j].bool()
                        record=dict(metrics=metrics(out[j].cpu().numpy(),y[j].cpu().numpy()),
                            fixed_fp=int((b & ~truth & ~q).sum()),fixed_fn=int((~b & truth & q).sum()),
                            new_fp=int((~b & ~truth & q).sum()),new_fn=int((b & truth & ~q).sum()),
                            mean_abs_delta=float(delta[j].abs().mean()),changed_pixels=int((q!=b).sum()),
                            top_k=k,topk_errors=int(err[j].flatten()[ranked[j]].sum()),
                            uncertainty_topk_errors=int(err[j].flatten()[uncertain[j]].sum()))
                        batch[j]['conditions'][key]=record
                        if i in selection['original32'] and condition=='correct':
                            expected=next(r for r in original_d['adamw/'+v]['records'] if r['index']==i)
                            assert record['metrics']==expected['output'],(v,i,'D exact reproduction failed')
                            assert batch[j]['baseline']==expected['baseline']
            semantic_ix=[i for i in ix if i in M['semantic_changed_cached_indices']]
            if semantic_ix:
                ins,zs,ys=data(semantic_ix,True)
                source=[lookup[rows[i]['source_text']] for i in semantic_ix]
                sin=(ins[0],embeddings[source],masks[source],ins[3],ins[4],ins[5])
                for v in ('T1','T2','T3','T4','template'):
                    ps=prediction(zs,heads[v](*sin,v),v)
                    for j,i in enumerate(semantic_ix):
                        next(r for r in batch if r['index']==i)['conditions'][v+'/semantic_source']=dict(metrics=metrics(ps[j].cpu().numpy(),ys[j].cpu().numpy()))
        e1.extend(batch)
    write_json(args.output/'e1_per_image.json',e1)
    write_json(args.output/'gradients_per_image.json',gradients)
    write_json(args.output/'matched_batch_loss.json',matched)
    write_json(args.output/'provenance.json',dict(source_git_commit=args.source_sha,manifest=M,
        cache_sha256_verified=True,checkpoint_sha256_verified=True,original32_all6_heads_exact_reproduction=True,
        train_updates=0,torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(),
        internal_holdout_accessed=False,official_validation_accessed=False,test_split_accessed=False,
        semantic_control_changed_n=len(M['semantic_changed_cached_indices'])))
    print(json.dumps(dict(event='e1_complete',n=len(e1),gradient_records=len(gradients))),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--source-sha',required=True);args=ap.parse_args()
    args.output.mkdir(exist_ok=False);runtime=dict(started_unix=time.time(),source_git_commit=args.source_sha,phase='running',train_updates=0)
    write_json(args.output/'runtime.json',runtime)
    try:
        main(args);runtime.update(phase='complete',completed_unix=time.time())
        runtime['artifacts']={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in args.output.iterdir() if p.name!='runtime.json'}
    except BaseException:
        runtime.update(phase='failed',failed_unix=time.time(),error=traceback.format_exc());traceback.print_exc()
    write_json(args.output/'runtime.json',runtime)
    if runtime['phase']=='failed':sys.exit(1)

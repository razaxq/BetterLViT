"""Train-only causal optimizer control: no old holdout or official Val/Test inference."""
import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
import sys
import time
import traceback
os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='1219')
HERE=Path(__file__).resolve().parent
M=json.loads((HERE/'manifest.json').read_text())
B=Path(M['completed_b_directory'])


def guard(event,args):
    if event in ('open','os.listdir','os.scandir') and args and isinstance(args[0],(str,bytes,os.PathLike)):
        parts=[p.lower() for p in Path(os.fsdecode(args[0])).parts]
        if any(p in parts for p in ('val_folder','test_folder','val_text.xlsx','test_text.xlsx')):
            raise RuntimeError('Val/Test prohibited in optimizer diagnosis')


sys.addaudithook(guard)
sys.path[:0]=[str(HERE),'/root/BetterLViT-recipe-r2']
import numpy as np
import torch
from heads import ResidualHead,VARIANTS,prediction
from analysis import digest,write_json,metrics
from utils import WeightedDiceFocal


def state_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main(args,runtime):
    torch.set_num_threads(4);torch.manual_seed(1219);np.random.seed(1219)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for relative,expected in M['b_artifact_sha256'].items():assert digest(B/relative)==expected,relative
    baseline=Path('/root/BetterLViT-recipe-r2')
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=baseline,text=True).strip()==M['baseline_source_git_commit']
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=baseline,text=True).strip()
    assert digest(HERE/'heads.py')==M['heads_sha256']
    rows=json.loads((B/'split.json').read_text())['records']
    order=json.loads((B/'results/training_order.json').read_text())['indices'][:M['steps']*16]
    assert len(order)==M['steps']*16
    assert all(rows[i]['eligible'] and rows[i]['partition']=='fit' for i in order)
    reference=json.loads((B/'results/train_diagnostics.json').read_text())
    diagnostic=[r['index'] for r in reference[0]['records']]
    assert len(diagnostic)==32 and all(rows[i]['partition']=='fit' for i in diagnostic)
    cache=json.loads((B/'results/cache_manifest.json').read_text())
    arrays={k:np.memmap(B/'cache'/(k+'.bin'),dtype=dtype,mode='r',shape=tuple(shape)) for k,(shape,dtype) in cache['shapes'].items()}
    for name,value in cache['files'].items():assert digest(B/'cache'/name)==value['sha256'],name
    text=np.load(B/'cache/text.npz');embeddings=torch.from_numpy(text['embeddings']).cuda();masks=torch.from_numpy(text['masks']).cuda()
    lookup={text:i for i,text in enumerate(cache['unique_texts'])}
    def batch_at(ix):
        assert all(rows[i]['partition']=='fit' and rows[i]['eligible'] for i in ix)
        feature=torch.from_numpy(np.array(arrays['features'][ix])).cuda().float()
        logits=torch.from_numpy(np.array(arrays['logits'][ix])).cuda()
        labels=np.unpackbits(np.array(arrays['masks'][ix]),axis=1,count=224*224,bitorder='little').reshape(-1,1,224,224)
        y=torch.from_numpy(labels).cuda().float()
        correct=[lookup[rows[i]['canonical']] for i in ix]; ref=[lookup[rows[i]['reference']] for i in ix]
        return (feature,embeddings[correct],masks[correct],embeddings[ref],masks[ref],torch.ones(len(ix),device='cuda',dtype=torch.bool)),logits,y
    torch.manual_seed(1219);initial=ResidualHead().cuda()
    assert state_hash(initial)==M['initial_state_sha256']
    cases={}
    for treatment in M['treatments']:
        for variant in VARIANTS:
            key=treatment+'/'+variant;head=copy.deepcopy(initial)
            optimizer=(torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001) if treatment=='adamw'
                else torch.optim.Adam(head.parameters(),lr=.001,weight_decay=.0001 if treatment=='adam_l2' else 0.))
            cases[key]=(head,optimizer,variant)
    initial_norms={n:float(p.detach().norm()) for n,p in initial.named_parameters()}
    del initial
    criterion=WeightedDiceFocal();history=[];telemetry=[]
    def assess(step):
        output={}
        with torch.no_grad():
            for key,(head,optimizer,variant) in cases.items():
                observations=[]
                for start in range(0,32,16):
                    ix=diagnostic[start:start+16];inputs,z,y=batch_at(ix)
                    delta=head(*inputs,variant);p=prediction(z,delta,variant)
                    probabilities=p.cpu().numpy();labels=y.cpu().numpy();base=z.sigmoid().cpu().numpy()
                    for j,i in enumerate(ix):
                        observations.append(dict(index=i,name=rows[i]['name'],baseline=metrics(base[j],labels[j]),
                            output=metrics(probabilities[j],labels[j]),mean_abs_delta=float(delta[j].abs().mean()),
                            changed_pixels=int(((probabilities[j]>.5)!=(base[j]>.5)).sum())))
                output[key]=dict(records=observations,parameter_norms={n:float(p.norm()) for n,p in head.named_parameters()})
        telemetry.append(dict(step=step,cases=output))
        if step in (0,256,512):
            original=next(x for x in reference if x['step']==step)['records']
            for variant in VARIANTS:
                for got,expected in zip(output['adam_l2/'+variant]['records'],original):
                    assert got['name']==expected['name']
                    assert got['output']==expected['outputs'][variant],(step,variant,got['name'],'Original Adam does not exactly reproduce B telemetry')
        write_json(args.output/'train_diagnostics.json',telemetry)
    assess(0)
    timings=[];runtime.update(phase='training',steps_total=M['steps'],steps_completed=0)
    write_json(args.output/'runtime.json',runtime)
    for step in range(1,M['steps']+1):
        started=time.time();inputs,z,y=batch_at(order[(step-1)*16:step*16])
        lr=.00001+.5*(.001-.00001)*(1+math.cos(math.pi*(step-1)/1023))
        losses={};gradients={}
        for key,(head,optimizer,variant) in cases.items():
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            loss=criterion(prediction(z,head(*inputs,variant),variant),y)
            assert torch.isfinite(loss),key
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
            optimizer.step();losses[key]=float(loss.detach());gradients[key]=float(norm)
        torch.cuda.synchronize();timings.append(time.time()-started)
        history.append(dict(step=step,lr=lr,loss=losses,gradient_norm=gradients,seconds=timings[-1]))
        if step in (64,256,512):assess(step)
        if step%32==0:
            mean=float(np.mean(timings[-128:]))
            runtime.update(steps_completed=step,mean_step_seconds=mean,
                expected_completion_unix=time.time()+(512-step)*mean*1.1+60)
            write_json(args.output/'runtime.json',runtime)
            print(json.dumps(dict(step=step,mean_step_seconds=mean,expected_completion_unix=runtime['expected_completion_unix'])),flush=True)
    runtime['training_completed_unix']=time.time()
    write_json(args.output/'history.json',history)
    final=telemetry[-1]['cases'];summary={}
    for key,value in final.items():
        rs=value['records'];summary[key]=dict(train_diagnostic_iou=float(np.mean([r['output']['iou'] for r in rs])),
            delta_iou=float(np.mean([r['output']['iou']-r['baseline']['iou'] for r in rs])),
            mean_abs_delta=float(np.mean([r['mean_abs_delta'] for r in rs])),
            changed_images=sum(r['changed_pixels']>0 for r in rs),parameter_norms=value['parameter_norms'],
            last_task_gradient_norm=history[-1]['gradient_norm'][key])
    write_json(args.output/'summary.json',dict(kind='train_only_optimizer_collapse_diagnostic',cases=summary,
        steps_per_case=512,source_git_commit=args.source_sha,initial_parameter_norms=initial_norms,
        original_adam_0_256_512_exact_reproduction=True,internal_holdout_evaluated=False,
        official_validation_accessed=False,test_split_accessed=False,architecture_gain_proven=False))
    torch.save(dict(source_git_commit=args.source_sha,manifest=M,steps=512,
        heads={key:{n:v.cpu() for n,v in h.state_dict().items()} for key,(h,_,_) in cases.items()}),args.output/'heads.pt')
    write_json(args.output/'provenance.json',dict(source_git_commit=args.source_sha,
        completed_b_source_git_commit=M['completed_b_source_git_commit'],manifest_sha256=digest(HERE/'manifest.json'),
        torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(),shared_cache_read_only=True,
        source_indices_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(),
        steps_per_case=512,internal_holdout_evaluated=False,official_validation_accessed=False,test_split_accessed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--source-sha',required=True);args=p.parse_args()
    args.output.mkdir(exist_ok=False);runtime=dict(phase='preflight',source_git_commit=args.source_sha,started_unix=time.time())
    write_json(args.output/'runtime.json',runtime)
    try:
        main(args,runtime);runtime.update(phase='complete',completed_unix=time.time())
        runtime['artifacts']={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in args.output.iterdir() if p.name!='runtime.json'}
    except BaseException:
        runtime.update(phase='failed',failed_unix=time.time(),error=traceback.format_exc());traceback.print_exc()
    write_json(args.output/'runtime.json',runtime)
    if runtime['phase']=='failed':sys.exit(1)

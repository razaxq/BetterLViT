"""Freeze R2, cache only Train, fit six heads identically, assess final holdout once."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
M=json.loads((HERE/'manifest.json').read_text())
ROOT=Path(M['baseline_repository'])
os.environ.update(BETTERLVIT_EXPERIMENT='r2_single_cosine',TEST_SPLIT_ALLOWED='0',AUTO_TEST_EVALUATE='0',
    HF_HOME='/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface',HF_HUB_OFFLINE='1',
    TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='1219',
    TOKENIZERS_PARALLELISM='false',BETTERLVIT_EPOCHS='80',BETTERLVIT_BATCH_SIZE='16')


def guard(event,args):
    if event in ('open','os.listdir','os.scandir') and args and isinstance(args[0],(str,bytes,os.PathLike)):
        parts=[p.lower() for p in Path(os.fsdecode(args[0])).parts]
        if any(p in parts for p in ('test_folder','val_folder','test_text.xlsx','val_text.xlsx')):
            raise RuntimeError('Official Val/Test access prohibited in phase B')


sys.addaudithook(guard)
sys.path[:0]=[str(ROOT),str(ROOT/'tools')];os.chdir(ROOT)
import numpy as np
import torch
from torch.utils.data import DataLoader
import Config as config
from Load_Dataset import ImageToImage2D,ValGenerator,_build_tokenizer
from export_validation_metrics import build_model
from utils import read_text,WeightedDiceFocal
sys.path.insert(0,str(HERE))
from analysis import digest,write_json,metrics
from split_policy import records as make_records
from heads import ResidualHead,VARIANTS,PROJECTED,prediction
from check_heads import run as check_heads
from screen_analysis import summarize


def state_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main(args,runtime):
    torch.manual_seed(M['seed']);np.random.seed(M['seed']);torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert digest(HERE/'split.json')==M['split_sha256']
    assert digest(HERE/'text_policy.py')==M['registered_text_policy_sha256']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==M['baseline_source_git_commit']
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()
    assert not config.text_use_lora and config.boundary_loss_weight==0
    assert digest(M['baseline_checkpoint'])==M['baseline_checkpoint_sha256']
    assert not (HERE/'cache').exists(),'Never reuse or overwrite an incomplete cache'
    cache=HERE/'cache';cache.mkdir()
    free=shutil.disk_usage(cache).free
    assert free>=M['cache_budget_bytes']+M['minimum_free_after_cache_bytes'],free
    proof=check_heads('cuda');write_json(args.output/'preflight.json',proof)
    checkpoint=torch.load(M['baseline_checkpoint'],map_location='cpu',weights_only=True)
    assert checkpoint['source_git_commit']==M['baseline_source_git_commit']
    assert checkpoint['epoch']==M['baseline_checkpoint_epoch_zero_based']
    model=build_model();model.load_state_dict(checkpoint['state_dict'],strict=True);del checkpoint
    model.requires_grad_(False);model.cuda().eval();before=state_hash(model)
    workbook=Path(config.task_dataset)/'Train_Val_text.xlsx'
    assert digest(workbook)==M['train_workbook_sha256']
    texts=read_text(str(workbook));tokenizer=_build_tokenizer()
    dataset=ImageToImage2D(config.train_dataset,config.task_name,texts,ValGenerator([224,224]),image_size=224)
    assert len(dataset)==M['train_count'] and Path(dataset.dataset_path).name=='Train_Folder'
    assert hashlib.sha256(('\n'.join(dataset.mask_list)+'\n').encode()).hexdigest()==M['train_membership_sha256']
    rows=make_records(dataset.mask_list,texts)
    assert rows==json.loads((HERE/'split.json').read_text())['records']
    encoded=tokenizer([r['source_text'] for r in rows],max_length=32,padding='max_length',truncation=True,return_tensors='pt')
    assert torch.equal(encoded['input_ids'],dataset.input_ids) and torch.equal(encoded['attention_mask'],dataset.attention_masks)
    shapes={'features':(M['feature_shape'],'float16'),'logits':(M['logits_shape'],'float32'),'masks':(M['mask_shape'],'uint8')}
    arrays={k:np.memmap(cache/(k+'.bin'),dtype=dtype,mode='w+',shape=tuple(shape)) for k,(shape,dtype) in shapes.items()}
    captured={}
    handles=[model.up1.register_forward_hook(lambda module,inputs,out:captured.update(feature=out)),
             model.outc.register_forward_hook(lambda module,inputs,out:captured.update(logits=out))]
    runtime.update(phase='caching_train',cache_total=len(rows));write_json(args.output/'runtime.json',runtime)
    offset=0;cache_started=time.time()
    with torch.inference_mode():
        for batch,names in DataLoader(dataset,batch_size=M['cache_batch_size'],shuffle=False,num_workers=0):
            count=len(names);assert list(names)==[r['name'] for r in rows[offset:offset+count]]
            probability=model(batch['image'].cuda(),batch['input_ids'].cuda(),batch['attention_mask'].cuda())
            z=captured['logits']; assert z.shape==(count,1,224,224)
            assert torch.equal(z.sigmoid(),probability),'Frozen logits do not reproduce native R2'
            feature=torch.nn.functional.avg_pool2d(captured['feature'],8)
            assert feature.shape==(count,64,28,28)
            arrays['features'][offset:offset+count]=feature.half().cpu().numpy()
            arrays['logits'][offset:offset+count]=z.cpu().numpy()
            label=batch['label'].numpy()>0
            arrays['masks'][offset:offset+count]=np.packbits(label.reshape(count,-1),axis=1,bitorder='little')
            offset+=count
            if offset%256==0:
                runtime.update(cache_completed=offset,cache_elapsed_seconds=time.time()-cache_started)
                write_json(args.output/'runtime.json',runtime)
        unique=sorted({r[key] for r in rows for key in ('canonical','reference')})
        tok=tokenizer(unique,max_length=32,padding='max_length',truncation=True,return_tensors='pt')
        assert max(len(x) for x in tokenizer(unique,truncation=False)['input_ids'])<=32
        embeddings=[]
        for start in range(0,len(unique),32):
            embeddings.append(model.encode_text(tok['input_ids'][start:start+32].cuda(),tok['attention_mask'][start:start+32].cuda()).cpu())
        embeddings=torch.cat(embeddings).numpy();masks=tok['attention_mask'].numpy()
        np.savez(cache/'text.npz',embeddings=embeddings,masks=masks)
    for handle in handles:handle.remove()
    assert state_hash(model)==before and all(p.grad is None for p in model.parameters())
    assert digest(M['baseline_checkpoint'])==M['baseline_checkpoint_sha256']
    for array in arrays.values():array.flush()
    assert sum(p.stat().st_size for p in cache.iterdir())<=M['cache_budget_bytes']
    assert shutil.disk_usage(cache).free>=M['minimum_free_after_cache_bytes']
    cache_manifest=dict(files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in cache.iterdir()},
        shapes=shapes,unique_texts=unique,baseline_state_sha256=before,
        native_probability_exact_all_train=True,baseline_unchanged=True,source_git_commit=args.source_sha,
        text_identity_verified_samples=len(rows),cache_elapsed_seconds=time.time()-cache_started,
        system_free_after_cache_bytes=shutil.disk_usage(cache).free)
    write_json(args.output/'cache_manifest.json',cache_manifest)
    del captured,model,dataset,arrays;torch.cuda.empty_cache()
    arrays={k:np.memmap(cache/(k+'.bin'),dtype=dtype,mode='r',shape=tuple(shape)) for k,(shape,dtype) in shapes.items()}
    text_lookup={text:i for i,text in enumerate(unique)}
    embeddings=torch.from_numpy(embeddings).cuda();masks=torch.from_numpy(masks).cuda()
    def batch_at(indices):
        selected=[rows[int(i)] for i in indices]
        feature=torch.from_numpy(np.array(arrays['features'][indices])).cuda().float()
        logits=torch.from_numpy(np.array(arrays['logits'][indices])).cuda()
        labels=np.unpackbits(np.array(arrays['masks'][indices]),axis=1,count=224*224,bitorder='little').reshape(-1,1,224,224)
        target=torch.from_numpy(labels).cuda().float()
        correct=[text_lookup[r['canonical']] for r in selected]; reference=[text_lookup[r['reference']] for r in selected]
        eligible=torch.tensor([r['eligible'] for r in selected],device='cuda')
        return feature,logits,target,embeddings[correct],masks[correct],embeddings[reference],masks[reference],eligible
    fit=[r['index'] for r in rows if r['partition']=='fit' and r['eligible']]
    held=[r['index'] for r in rows if r['partition']=='holdout']
    diagnostic=sorted(fit,key=lambda i:hashlib.sha256(('text-b-diagnostic:'+rows[i]['name']).encode()).hexdigest())[:32]
    fit_all=[r['index'] for r in rows if r['partition']=='fit']
    fit_areas=np.unpackbits(np.array(arrays['masks'][fit_all]),axis=1,count=224*224,bitorder='little').sum(1)
    small_cutoff=float(np.quantile(fit_areas,.25)); del fit_areas
    torch.manual_seed(M['seed']);initial=ResidualHead().cuda()
    heads={v:copy.deepcopy(initial) for v in VARIANTS}
    assert len({state_hash(h) for h in heads.values()})==1
    initial_hash=state_hash(initial); del initial
    optimizers={v:torch.optim.Adam(h.parameters(),lr=M['lr_start'],weight_decay=M['weight_decay']) for v,h in heads.items()}
    criterion=WeightedDiceFocal()
    assert config.dice_loss_weight==config.focal_loss_weight==.5 and config.focal_gamma==2
    telemetry=[]
    def evaluate(indices,wrong_text=False):
        output=[]
        with torch.no_grad():
            for start in range(0,len(indices),M['batch_size']):
                ix=indices[start:start+M['batch_size']]
                feature,z,y,text,mask,ref,refmask,eligible=batch_at(ix)
                base=z.sigmoid();ys=y.cpu().numpy();base_np=base.cpu().numpy()
                current=[dict(**rows[i],outputs={'baseline':metrics(base_np[j],ys[j])},diagnostics={}) for j,i in enumerate(ix)]
                for variant,head in heads.items():
                    delta=head(feature,text,mask,ref,refmask,eligible,variant);p=prediction(z,delta,variant)
                    values=p.cpu().numpy();deltas=delta.cpu().numpy()
                    errors=(p.double().sum((1,2,3))-base.double().sum((1,2,3))).abs().cpu().tolist()
                    if variant in PROJECTED:assert max(errors)<=M['maximum_mass_error_pixels'],(variant,max(errors))
                    for j,row in enumerate(current):
                        row['outputs'][variant]=metrics(values[j],ys[j])
                        row['diagnostics'][variant]=dict(mean_abs_delta=float(np.abs(deltas[j]).mean()),
                            maximum_abs_delta=float(np.abs(deltas[j]).max()),absolute_mass_error=errors[j],
                            probability_mae=float(np.abs(values[j]-base_np[j]).mean()),
                            changed_pixels=int(((values[j]>.5)!=(base_np[j]>.5)).sum()))
                    if wrong_text and variant in ('T1','T3','T4'):
                        wrong=prediction(z,head(feature,ref,refmask,text,mask,eligible,variant),variant).cpu().numpy()
                        for j,row in enumerate(current):row['outputs'][variant+'_wrong']=metrics(wrong[j],ys[j])
                output.extend(current)
        return output
    telemetry.append(dict(step=0,records=evaluate(diagnostic)))
    for row in telemetry[0]['records']:
        assert all(row['outputs'][v]==row['outputs']['baseline'] for v in VARIANTS)
    rng=np.random.default_rng(M['seed']);order=[]
    while len(order)<M['steps']*M['batch_size']:order.extend(rng.permutation(fit).tolist())
    order=order[:M['steps']*M['batch_size']]
    write_json(args.output/'training_order.json',dict(indices=order,initial_state_sha256=initial_hash,fit_indices=fit,holdout_indices=held))
    history=[];step_times=[];runtime.update(phase='training',steps_completed=0,steps_total=M['steps'])
    write_json(args.output/'runtime.json',runtime)
    for step in range(1,M['steps']+1):
        started=time.time();ix=order[(step-1)*M['batch_size']:step*M['batch_size']]
        feature,z,y,text,mask,ref,refmask,eligible=batch_at(ix);losses={};gradient_norms={}
        lr=M['lr_end']+.5*(M['lr_start']-M['lr_end'])*(1+math.cos(math.pi*(step-1)/(M['steps']-1)))
        for variant,head in heads.items():
            optimizer=optimizers[variant]
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            delta=head(feature,text,mask,ref,refmask,eligible,variant)
            p=prediction(z,delta,variant);loss=criterion(p,y)
            assert torch.isfinite(loss),variant
            loss.backward()
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in head.parameters())
            norm=torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True)
            optimizer.step();losses[variant]=float(loss.detach());gradient_norms[variant]=float(norm)
        torch.cuda.synchronize();step_times.append(time.time()-started)
        history.append(dict(step=step,lr=lr,loss=losses,gradient_norm=gradient_norms,seconds=step_times[-1]))
        if step in M['training_telemetry_steps']:telemetry.append(dict(step=step,records=evaluate(diagnostic)))
        if step%32==0 or step==1:
            remaining=(M['steps']-step)*float(np.mean(step_times[-128:]))
            runtime.update(steps_completed=step,last_step_unix=time.time(),mean_step_seconds=float(np.mean(step_times[-128:])),
                expected_completion_unix=time.time()+remaining*1.10+240)
            write_json(args.output/'runtime.json',runtime)
            print(json.dumps(dict(step=step,lr=lr,loss=losses,expected_completion_unix=runtime['expected_completion_unix'])),flush=True)
    runtime.update(phase='final_internal_holdout',training_completed_unix=time.time());write_json(args.output/'runtime.json',runtime)
    write_json(args.output/'history.json',history);write_json(args.output/'train_diagnostics.json',telemetry)
    torch.save(dict(source_git_commit=args.source_sha,baseline_source_git_commit=M['baseline_source_git_commit'],
        baseline_checkpoint_sha256=M['baseline_checkpoint_sha256'],steps=M['steps'],manifest=M,
        heads={v:{k:x.cpu() for k,x in h.state_dict().items()} for v,h in heads.items()}),args.output/'heads.pt')
    final=evaluate(held,wrong_text=True)
    write_json(args.output/'records.json',dict(source_git_commit=args.source_sha,records=final,
        final_step=M['steps'],small_cutoff_fit_gt_pixels=small_cutoff,official_validation_accessed=False,test_split_accessed=False))
    write_json(args.output/'summary.json',summarize(final,small_cutoff,M['mechanism_screen_gate']))
    write_json(args.output/'provenance.json',dict(source_git_commit=args.source_sha,manifest_sha256=digest(HERE/'manifest.json'),
        baseline_source_git_commit=M['baseline_source_git_commit'],baseline_checkpoint_sha256=digest(M['baseline_checkpoint']),
        baseline_state_sha256=before,baseline_state_unchanged=True,torch=torch.__version__,numpy=np.__version__,
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),final_step=M['steps'],
        shared_fs_written=False,official_validation_accessed=False,test_split_accessed=False,
        initial_state_sha256=initial_hash,final_state_sha256={v:state_hash(h) for v,h in heads.items()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--source-sha',required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    runtime=dict(phase='preflight',source_git_commit=args.source_sha,started_unix=time.time())
    write_json(args.output/'runtime.json',runtime)
    try:
        main(args,runtime)
        runtime.update(phase='complete',completed_unix=time.time())
        runtime['artifacts']={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in args.output.iterdir() if p.name!='runtime.json'}
    except BaseException:
        runtime.update(phase='failed',failed_unix=time.time(),error=traceback.format_exc());traceback.print_exc()
    write_json(args.output/'runtime.json',runtime)
    if runtime['phase']=='failed':sys.exit(1)

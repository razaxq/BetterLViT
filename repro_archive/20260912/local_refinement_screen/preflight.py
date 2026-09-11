"""Reproduce the old frozen R2 cache, save only fit point features, time fresh heads."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import time
import traceback
from common import *
from check_refiner import run as synthetic_checks

def main(args,runtime):
    setup();data=Inputs();checks={d:synthetic_checks(d) for d in ('cpu','cuda')}
    write_json(args.output/'synthetic_checks.json',checks)
    assert shutil.disk_usage(HERE).free>=M['exact_cache_bytes']+M['minimum_free_bytes']
    assert digest(M['baseline_checkpoint'])==M['baseline_checkpoint_sha256']
    import Config as config
    from Load_Dataset import ImageToImage2D,ValGenerator,_build_tokenizer
    from export_validation_metrics import build_model
    from utils import read_text
    from torch.utils.data import DataLoader
    os.chdir(ROOT)
    assert not config.text_use_lora and config.boundary_loss_weight==0
    workbook=Path(config.task_dataset)/'Train_Val_text.xlsx';assert digest(workbook)==M['train_workbook_sha256']
    texts=read_text(str(workbook))
    dataset=ImageToImage2D(config.train_dataset,config.task_name,texts,ValGenerator([224,224]),image_size=224)
    assert len(dataset)==5716 and Path(dataset.dataset_path).name=='Train_Folder'
    assert hashlib.sha256(('\n'.join(dataset.mask_list)+'\n').encode()).hexdigest()==M['train_membership_sha256']
    for r in data.rows:
        assert dataset.mask_list[r['index']]==r['mask_name'] and texts[r['mask_name']]==r['source_text']
    tok=_build_tokenizer()([r['source_text'] for r in data.rows],max_length=32,padding='max_length',truncation=True,return_tensors='pt')
    fit=[r['index'] for r in data.rows]
    assert torch.equal(tok['input_ids'],dataset.input_ids[fit]) and torch.equal(tok['attention_mask'],dataset.attention_masks[fit])
    batches=[];counts=[]
    for start in range(0,5716,16):
        physical=min(16,5716-start);ix=[i for i in range(start,start+physical) if i in data.by_index]
        if not ix:continue
        counts.append(len(ix));batches.append(ix+[ix[0]]*(physical-len(ix)))
    assert all(i in data.by_index for batch in batches for i in batch)
    checkpoint=torch.load(M['baseline_checkpoint'],map_location='cpu',weights_only=True)
    assert checkpoint['source_git_commit']==M['baseline_source_git_commit'] and checkpoint['epoch']==M['baseline_checkpoint_epoch_zero_based']
    model=build_model();model.load_state_dict(checkpoint['state_dict'],strict=True);del checkpoint
    model.requires_grad_(False);model.cuda().eval();before=state_hash(model)
    assert before==data.manifest['baseline_state_sha256']
    cache=HERE/'cache';cache.mkdir(exist_ok=False)
    fine=np.memmap(cache/'fine.bin',mode='w+',dtype='float16',shape=tuple(M['feature_cache_shape']))
    indices=np.memmap(cache/'indices.bin',mode='w+',dtype='uint16',shape=tuple(M['point_index_shape']))
    captured={};handles=[model.up1.register_forward_hook(lambda module,inputs,out:captured.update(feature=out)),
        model.outc.register_forward_hook(lambda module,inputs,out:captured.update(logits=out))]
    seen=[];began=time.time();runtime.update(phase='caching_fit');write_json(args.output/'runtime.json',runtime)
    with torch.inference_mode():
        for b,(batch,names) in enumerate(DataLoader(dataset,batch_sampler=batches,num_workers=0)):
            valid=counts[b];ix=batches[b][:valid]
            assert list(names)==[data.old[i]['name'] for i in batches[b]]
            p=model(batch['image'].cuda(),batch['input_ids'].cuda(),batch['attention_mask'].cuda())
            z=captured['logits'];feature=captured['feature'];assert torch.equal(z.sigmoid(),p)
            coarse=torch.nn.functional.avg_pool2d(feature,8).half()
            _,old_z,y=data.old_batch(ix)
            assert torch.equal(z[:valid],old_z),'Old logits identity failed; no tolerance relaxation'
            assert torch.equal(coarse[:valid].cpu(),torch.from_numpy(np.array(data.arrays['features'][ix])))
            assert torch.equal((batch['label'][:valid].cuda()>0),y.bool())
            uncertainty=old_z.sigmoid()*(1-old_z.sigmoid())
            point=torch.argsort(uncertainty.flatten(1),dim=1,descending=True,stable=True)[:,:1024]
            assert all(len(torch.unique(row))==1024 for row in point)
            values=feature[:valid].flatten(2).transpose(1,2).gather(1,point[:,:,None].expand(-1,-1,64)).half().cpu().numpy()
            cr=[data.by_index[i]['cache_row'] for i in ix];fine[cr]=values;indices[cr]=point.cpu().numpy().astype(np.uint16)
            seen.extend(ix)
    for h in handles:h.remove()
    fine.flush();indices.flush()
    assert sorted(seen)==fit and state_hash(model)==before and all(p.grad is None for p in model.parameters())
    assert sum(p.stat().st_size for p in cache.iterdir())==M['exact_cache_bytes']
    assert shutil.disk_usage(cache).free>=M['minimum_free_bytes']
    cache_seconds=time.time()-began
    manifest=dict(source_git_commit=args.source_sha,baseline_unchanged=True,baseline_state_sha256=before,
        exact_identity_verified_samples=len(seen),old_b_holdout_images_loaded=0,original_batch_shape_preserved=True,
        exact_cache_bytes=M['exact_cache_bytes'],cache_seconds=cache_seconds,
        files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in cache.iterdir()})
    write_json(args.output/'cache_manifest.json',manifest)
    del captured,model,dataset,tok,fine,indices

def finish_benchmark(args,runtime):
    torch.cuda.empty_cache();data=Inputs(fine=True)
    ix=[r['index'] for r in data.rows if r['fold']!=0][:16];inputs,y=data.batch(ix,training_fold=0)
    torch.manual_seed(1219);initial=LocalRefiner().cuda();initial_hash=state_hash(initial)
    heads={v:copy.deepcopy(initial) for v in VARIANTS}
    optimizers={v:torch.optim.AdamW(h.parameters(),lr=.001,weight_decay=.0001) for v,h in heads.items()}
    criterion=WeightedDiceFocal()
    with torch.no_grad():
        for v,h in heads.items():assert torch.equal(predict(inputs[2],h(*inputs,v),inputs[3],v),inputs[2].sigmoid())
    times=[];last_gradients={}
    for step in range(24):
        began=time.time()
        for v,h in heads.items():
            optimizers[v].zero_grad(set_to_none=True)
            loss=criterion(predict(inputs[2],h(*inputs,v),inputs[3],v),y);assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(h.parameters(),5.,error_if_nonfinite=True)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in h.parameters())
            last_gradients[v]=float(h.fc1.weight.grad.norm());optimizers[v].step()
        torch.cuda.synchronize();times.append(time.time()-began)
    assert all(g>0 for g in last_gradients.values())
    before=time.time();observations=observe(heads,inputs,y,[data.by_index[i] for i in ix]);torch.cuda.synchronize()
    eval_batch_seconds=time.time()-before
    mean=float(np.mean(times[8:]));expected=mean*2048*5*1.3+eval_batch_seconds*(4585/16)*1.3+120
    write_json(args.output/'benchmark.json',dict(passed=True,source_git_commit=args.source_sha,
        initial_state_sha256=initial_hash,parameters_per_head=sum(p.numel() for p in initial.parameters()),
        real_fit_batch=ix,discarded_preflight_updates=24,mean_four_head_step_seconds=mean,
        four_head_evaluation_batch_seconds=eval_batch_seconds,predicted_train_and_oof_seconds=expected,
        last_fc1_gradient_norm=last_gradients,all_preflight_heads_discarded=True,
        torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(),
        maximum_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),system_free_bytes=shutil.disk_usage(HERE).free))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--source-sha',required=True);args=ap.parse_args()
    args.output.mkdir(exist_ok=False);runtime=dict(phase='preflight',source_git_commit=args.source_sha,started_unix=time.time())
    write_json(args.output/'runtime.json',runtime)
    try:
        main(args,runtime);finish_benchmark(args,runtime);runtime.update(phase='complete',completed_unix=time.time())
        runtime['artifacts']={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in args.output.iterdir() if p.name!='runtime.json'}
    except BaseException:
        runtime.update(phase='failed',failed_unix=time.time(),error=traceback.format_exc());traceback.print_exc()
    write_json(args.output/'runtime.json',runtime)
    if runtime['phase']=='failed':sys.exit(1)

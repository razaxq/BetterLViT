"""Fit four matched heads per fold, then evaluate each own held fold exactly once."""
import argparse
import copy
import json
import math
from pathlib import Path
import time
import traceback
from common import *

def save_checkpoint(path,value):
    temporary=path.with_suffix('.tmp');torch.save(value,temporary);temporary.replace(path)

def main(args,runtime):
    setup();data=Inputs(fine=True)
    benchmark=json.loads((HERE/'preflight/benchmark.json').read_text());assert benchmark['passed']
    assert benchmark['source_git_commit']==args.source_sha
    torch.manual_seed(1219);initial=LocalRefiner().cuda()
    assert state_hash(initial)==benchmark['initial_state_sha256']
    criterion=WeightedDiceFocal();all_records=[];all_history=[];all_orders={};case_hashes={}
    global_times=[];evaluation_seconds=0.;completed_steps=0
    initial_hash=state_hash(initial)
    for fold in range(5):
        fit=[r['index'] for r in data.rows if r['fold']!=fold]
        held=[r['index'] for r in data.rows if r['fold']==fold]
        assert set(fit).isdisjoint(held) and len(fit)+len(held)==4585
        assert {data.by_index[i]['group_id'] for i in fit}.isdisjoint({data.by_index[i]['group_id'] for i in held})
        cutoff=data.train_small_cutoff(fold)
        rng=np.random.default_rng(1219+fold);order=[]
        while len(order)<M['steps_per_fold']*16:order.extend(rng.permutation(fit).tolist())
        order=order[:M['steps_per_fold']*16];all_orders[str(fold)]=dict(indices=order,fit=fit,held=held,small_cutoff_fit_only=cutoff)
        heads={v:copy.deepcopy(initial) for v in VARIANTS}
        assert all(state_hash(h)==initial_hash for h in heads.values())
        optimizers={v:torch.optim.AdamW(h.parameters(),lr=.001,weight_decay=.0001) for v,h in heads.items()}
        history=[]
        # Initial identity is checked only on this fold's optimization data.
        inputs,y=data.batch(fit[:16],training_fold=fold)
        with torch.no_grad():
            for v,h in heads.items():assert torch.equal(predict(inputs[2],h(*inputs,v),inputs[3],v),inputs[2].sigmoid())
        runtime.update(phase='training',fold=fold,steps_in_fold=0);write_json(args.output/'runtime.json',runtime)
        for step in range(1,M['steps_per_fold']+1):
            began=time.time();ix=order[(step-1)*16:step*16];inputs,y=data.batch(ix,training_fold=fold)
            lr=.00001+.5*(.001-.00001)*(1+math.cos(math.pi*(step-1)/(M['steps_per_fold']-1)))
            losses={};gradients={}
            for v,h in heads.items():
                optimizer=optimizers[v]
                for group in optimizer.param_groups:group['lr']=lr
                optimizer.zero_grad(set_to_none=True)
                delta=h(*inputs,v);p=predict(inputs[2],delta,inputs[3],v);loss=criterion(p,y)
                assert torch.isfinite(loss),v
                loss.backward();assert all(param.grad is not None and torch.isfinite(param.grad).all() for param in h.parameters())
                norm=torch.nn.utils.clip_grad_norm_(h.parameters(),5.,error_if_nonfinite=True)
                optimizer.step();losses[v]=float(loss.detach());gradients[v]=float(norm)
            torch.cuda.synchronize();seconds=time.time()-began;global_times.append(seconds);completed_steps+=1
            history.append(dict(step=step,lr=lr,loss=losses,gradient_norm=gradients,seconds=seconds))
            if step in M['training_checkpoints']:
                save_checkpoint(args.output/'resume.pt',dict(source_git_commit=args.source_sha,manifest=M,fold=fold,step=step,
                    heads={v:h.state_dict() for v,h in heads.items()},optimizers={v:o.state_dict() for v,o in optimizers.items()},
                    torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all()))
            if step%128==0:
                mean=float(np.mean(global_times[-256:]));remaining=5*2048-completed_steps
                # All remaining OOF inference cost plus a fixed summary margin.
                remaining_images=sum(r['fold']>=fold for r in data.rows)
                eta=time.time()+remaining*mean*1.15+remaining_images/16*benchmark['four_head_evaluation_batch_seconds']*1.25+120
                runtime.update(steps_in_fold=step,completed_steps=completed_steps,mean_four_head_step_seconds=mean,expected_completion_unix=eta)
                write_json(args.output/'runtime.json',runtime)
                print(json.dumps(dict(fold=fold,step=step,total_steps=completed_steps,expected_completion_unix=eta)),flush=True)
        fold_finished=time.time();runtime.update(phase='fold_oof',fold_training_completed_unix=fold_finished)
        if fold==4:runtime['training_completed_unix']=fold_finished
        write_json(args.output/'runtime.json',runtime)
        states={v:{n:t.detach().cpu().clone() for n,t in h.state_dict().items()} for v,h in heads.items()}
        save_checkpoint(args.output/f'fold_{fold}_heads.pt',dict(source_git_commit=args.source_sha,manifest=M,fold=fold,
            steps=2048,initial_state_sha256=initial_hash,fit_indices=fit,held_indices=held,heads=states))
        case_hashes[str(fold)]={v:state_hash(h) for v,h in heads.items()}
        records=[];eval_start=time.time()
        for start in range(0,len(held),16):
            ix=held[start:start+16];assert all(data.by_index[i]['fold']==fold for i in ix)
            inputs,y=data.batch(ix)
            batch=observe(heads,inputs,y,[data.by_index[i] for i in ix])
            for r in batch:r['small_by_fit_cutoff']=r['outputs']['baseline']['gt_area']<=cutoff
            records.extend(batch)
        evaluation_seconds+=time.time()-eval_start
        write_json(args.output/f'fold_{fold}_records.json',dict(source_git_commit=args.source_sha,fold=fold,steps=2048,
            held_evaluations=1,small_cutoff_fit_only=cutoff,training_completed_unix=fold_finished,records=records))
        all_records.extend(records);all_history.append(dict(fold=fold,history=history))
        del heads,optimizers,states;torch.cuda.empty_cache()
    assert len(all_records)==4585 and len({r['index'] for r in all_records})==4585
    assert state_hash(initial)==initial_hash
    write_json(args.output/'records.json',dict(source_git_commit=args.source_sha,records=sorted(all_records,key=lambda r:r['index']),
        kind='original_B_fit_head_crossfit_not_independent_of_R2',official_validation_accessed=False,test_split_accessed=False))
    write_json(args.output/'history.json',all_history);write_json(args.output/'training_orders.json',all_orders)
    write_json(args.output/'provenance.json',dict(source_git_commit=args.source_sha,manifest_sha256=digest(HERE/'manifest.json'),
        baseline_source_git_commit=M['baseline_source_git_commit'],baseline_checkpoint_sha256=M['baseline_checkpoint_sha256'],
        baseline_cache_read_only=True,train_updates_per_head_per_fold=2048,folds=5,heads_per_fold=4,
        initial_state_sha256=initial_hash,final_state_sha256=case_hashes,old_b_holdout_accessed=False,
        official_validation_accessed=False,test_split_accessed=False,shared_fs_written=False,
        torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(),
        mean_four_head_step_seconds=float(np.mean(global_times)),total_oof_seconds=evaluation_seconds))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--source-sha',required=True);args=ap.parse_args()
    args.output.mkdir(exist_ok=False);runtime=dict(phase='preflight',source_git_commit=args.source_sha,started_unix=time.time(),total_steps=5*2048)
    write_json(args.output/'runtime.json',runtime)
    try:
        main(args,runtime);runtime.update(phase='complete',completed_unix=time.time())
        runtime['artifacts']={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in args.output.iterdir() if p.name!='runtime.json'}
    except BaseException:
        runtime.update(phase='failed',failed_unix=time.time(),error=traceback.format_exc());traceback.print_exc()
    write_json(args.output/'runtime.json',runtime)
    if runtime['phase']=='failed':sys.exit(1)

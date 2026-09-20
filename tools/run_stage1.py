"""One immutable full-module run: Train -> Val-selected Best -> fixed Test."""
import argparse,fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stage1_protocol import validate_manifest,environment

def write(path,value):
    temp=Path(str(path)+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1];m=validate_manifest(json.loads((repo/'experiment_manifests/active_stage1.json').read_text()))
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    assert subprocess.check_output(['git','rev-parse',m['experiment_tag']+'^{commit}'],cwd=repo,text=True).strip()==sha
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
    assert subprocess.run(['pgrep','-f','[t]rain_model.py'],capture_output=True).returncode!=0
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert shutil.disk_usage(repo).free>4*2**30
    args.run.mkdir(parents=True,exist_ok=False)
    lock=(args.run.parent/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=environment(m,sha,args.run);python=sys.executable
    state=dict(phase='training',source_git_commit=sha,manifest=m,repository=str(repo),started_unix=time.time(),test_split_accessed=False)
    write(args.run/'runtime.json',state)
    try:
        with (args.run/'training.log').open('w') as log:
            proc=subprocess.Popen([python,'-u','train_model.py'],cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            state['training_pid']=proc.pid;write(args.run/'runtime.json',state);rc=proc.wait()
        state.update(training_rc=rc,training_ended_unix=time.time(),training_pid=None)
        if rc:raise RuntimeError('Training failed: '+str(rc))
        bests=list((repo/'Covid19'/'BetterLViT'/m['profile']).glob('*/models/best_model-BetterLViT.pth.tar'))
        assert len(bests)==1,'Expected one Best in new worktree'
        best=bests[0];state['best_checkpoint']=str(best)
        with best.open('rb') as f:state['best_checkpoint_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
        for split,count in [('validation',1429),('test',2113)]:
            state.update(phase=split,test_split_accessed=split=='test');write(args.run/'runtime.json',state)
            command=[python,'tools/export_stage1_metrics.py','--experiment',m['profile'],'--checkpoint',str(best),
                '--output',str(args.run/(split+'.json')),'--batch-size','16','--threshold','0.5','--split',split]
            split_env=dict(env)
            if split=='test':
                command+=['--expected-best-epoch',str(state['selected_best_epoch'])];split_env['TEST_SPLIT_ALLOWED']='1'
            with (args.run/(split+'.log')).open('w') as log:
                rc=subprocess.run(command,cwd=repo,env=split_env,stdout=log,stderr=subprocess.STDOUT).returncode
            state[split+'_rc']=rc;state[split+'_ended_unix']=time.time()
            if rc:raise RuntimeError(split+' evaluation failed: '+str(rc))
            d=json.loads((args.run/(split+'.json')).read_text())
            assert d['samples']==count and d['checkpoint_git_commit']==sha and d['seed']==m['seed'] and d['threshold']==.5
            assert d['race_enabled']==m['race_enabled'] and d['race_route_enabled']==m['race_route_enabled'] and d['race_binding_repair']==m['race_binding_repair']
            assert d['decoder_fusion_mode']==m['decoder_fusion_mode']
            if split=='validation':state['selected_best_epoch']=d['checkpoint_best_epoch']
            else:assert d['checkpoint_best_epoch']==state['selected_best_epoch']
            state[split+'_metrics']={k:d[k] for k in ('macro_iou','macro_dice','macro_precision','macro_recall','macro_brier')}
        state.update(phase='complete',completed_unix=time.time())
    except Exception as e:
        state.update(phase='failed',error=str(e),failed_unix=time.time());raise
    finally:write(args.run/'runtime.json',state)

if __name__=='__main__':main()

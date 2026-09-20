"""Six fixed missing-cell runs, gated on prior queue success, with Train-only preflights."""
import argparse,fcntl,json,subprocess,sys,time,shutil
from pathlib import Path
from run_cos_restart import write
from cos_restart_protocol import validate_manifest,environment

def validate_spec(spec):
    assert spec['scope']=='cos_restart_overall_missing_cells'
    assert len(spec['runs'])==6 and len({r['source_git_commit'] for r in spec['runs']})==6
    assert [(r['configuration'],r['seed']) for r in spec['runs']]==[(g,s) for s in (1219,2027,3407) for g in ('CR1','CR2')]
    assert len(spec['dependency_runs'])==4
    assert {(r['configuration'],r['seed']) for r in spec['dependency_runs']}=={(g,s) for g in ('CR0','CR3') for s in (2027,3407)}

def dependency_ready(spec):
    q=json.loads(Path(spec['dependency_status']).read_text())
    if q['phase']=='failed':raise RuntimeError('Predecessor queue failed; refuse extension')
    if q['phase']!='complete':return False
    assert q['active'] is None and len(q['runs'])==4
    manifest=json.loads(Path(spec['dependency_manifest']).read_text())
    assert manifest['runs']==spec['dependency_runs'],'Predecessor manifest changed'
    for r in spec['dependency_runs']:
        done=next(x for x in q['runs'] if x['run_id']==r['run_id'])
        assert done['source_git_commit']==r['source_git_commit'] and done['returncode']==0 and done['phase']=='complete'
        rt=json.loads((Path(r['run'])/'runtime.json').read_text())
        assert rt['phase']=='complete' and rt['source_git_commit']==r['source_git_commit']
        assert rt['training_rc']==rt['validation_rc']==rt['test_rc']==0
        assert Path(rt['best_checkpoint']).is_file()
    return True

def idle():
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU in use'
    assert subprocess.run(['pgrep','-f','[t]rain_model.py'],capture_output=True).returncode!=0,'Training process still active'

def check_source(r):
    repo=Path(r['repository'])
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==r['source_git_commit']
    assert subprocess.check_output(['git','rev-parse',r['experiment_tag']+'^{commit}'],cwd=repo,text=True).strip()==r['source_git_commit']
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
    assert (repo/'Covid19').resolve()==Path(r['model_storage']).resolve()
    assert shutil.disk_usage(r['model_storage']).free>8*2**30
    assert not Path(r['run']).exists()

def preflight(r,stage):
    check_source(r);idle();repo=Path(r['repository'])
    m=validate_manifest(json.loads((repo/'experiment_manifests/active_cos_restart.json').read_text()))
    env=environment(m,r['source_git_commit'],Path(r['run']));records={}
    def execute(label,command,run_env):
        path=stage/'preflights'/(r['run_id']+'_'+label+'.log')
        with path.open('w') as log:rc=subprocess.run(command,cwd=repo,env=run_env,stdout=log,stderr=subprocess.STDOUT,timeout=300).returncode
        if rc:raise RuntimeError('Preflight failed: '+str(path)+' '+path.read_text()[-3000:])
        return json.loads(next(line for line in reversed(path.read_text().splitlines()) if line.startswith('{')))
    records['checks']=execute('checks',[sys.executable,'tools/check_cos_restart.py'],env)
    records['train_preflight']=execute('train_preflight',[sys.executable,'tools/preflight_stage1.py'],env)
    if r['configuration']=='CR2':
        reference=json.loads((Path('/root/autodl-tmp/stage1_overall_20260915/preflights')/('j2s'+str(r['seed'])+'.json')).read_text())['records']['train_preflight']
    else:
        reference_repo=Path('/root/autodl-tmp/BetterLViT-stage1-development')
        reference_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=reference_repo,text=True).strip()
        assert reference_sha=='014f42cf084dab01ab4e771a511efcec84276261'
        reference_env=dict(env,BETTERLVIT_EXPERIMENT='r2_single_cosine',BETTERLVIT_GIT_COMMIT=reference_sha)
        reference=execute('reference',[sys.executable,'tools/preflight_stage1.py','--reference-repo',str(reference_repo)],reference_env)
    assert all(reference[k]==records['train_preflight'][k] for k in ('initialization_sha256','output_sha256','loss')),'Same-seed reference mismatch'
    proof=dict(passed=True,source_git_commit=r['source_git_commit'],run_id=r['run_id'],records=records,reference=reference,historical_single_cosine_initialization_output_loss_identical=True,test_split_accessed=False)
    write(Path(r['preflight']),proof)
    idle()
    return proof

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);a=p.parse_args()
    spec=json.loads(a.manifest.read_text());stage=a.manifest.parent;validate_spec(spec)
    lock=(stage/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (stage/'queue_status.json').exists(),'Refuse duplicate queue'
    state=dict(phase='waiting_dependency',started_unix=time.time(),active=None,runs=[],queue_pid=__import__('os').getpid(),dependency_status=spec['dependency_status'])
    write(stage/'queue_status.json',state)
    try:
        while not dependency_ready(spec):time.sleep(60)
        idle();state.update(dependency_completed_unix=time.time(),phase='preflight');write(stage/'queue_status.json',state)
        for r in spec['runs']:
            state.update(phase='preflight',active=r['run_id'],runner_pid=None);write(stage/'queue_status.json',state)
            preflight(r,stage)
            state.update(phase='running');write(stage/'queue_status.json',state)
            with (stage/(r['run_id']+'_runner.log')).open('w') as log:
                proc=subprocess.Popen([sys.executable,'-u','tools/run_cos_restart.py','--run',r['run']],cwd=r['repository'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                state['runner_pid']=proc.pid;write(stage/'queue_status.json',state);rc=proc.wait()
            runtime=Path(r['run'])/'runtime.json';result=json.loads(runtime.read_text()) if runtime.exists() else {}
            state['runs'].append(dict(run_id=r['run_id'],returncode=rc,phase=result.get('phase','failed'),source_git_commit=r['source_git_commit'],run=r['run']))
            if rc or result.get('phase')!='complete':raise RuntimeError('Stopped at '+r['run_id'])
        state.update(phase='complete',active=None,runner_pid=None,completed_unix=time.time())
    except Exception as e:
        state.update(phase='failed',error=str(e),failed_unix=time.time());raise
    finally:write(stage/'queue_status.json',state)

if __name__=='__main__':main()

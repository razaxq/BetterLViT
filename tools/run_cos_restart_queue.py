"""Exactly four additional serial authorized runs; stop on failure, never auto-change config."""
import argparse,fcntl,json,subprocess,sys,time
from pathlib import Path
from run_cos_restart import write

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);a=p.parse_args()
    spec=json.loads(a.manifest.read_text());stage=a.manifest.parent
    assert len(spec['runs'])==4 and spec['scope']=='cos_restart_three_seed_extension'
    assert len({r['source_git_commit'] for r in spec['runs']})==4
    assert {(r['configuration'],r['seed']) for r in spec['runs']}=={(g,s) for g in ('CR0','CR3') for s in (2027,3407)}
    assert not (stage/'queue_status.json').exists(),'Refuse duplicate queue; inspect existing status'
    lock=(stage/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(phase='running',started_unix=time.time(),active=None,runs=[])
    try:
        for r in spec['runs']:
            repo=Path(r['repository']);run=Path(r['run']);proof=json.loads(Path(r['preflight']).read_text())
            assert proof['passed'] and proof['source_git_commit']==r['source_git_commit']
            assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==r['source_git_commit']
            assert not run.exists()
            state['active']=r['run_id'];write(stage/'queue_status.json',state)
            with (stage/(r['run_id']+'_runner.log')).open('w') as log:
                proc=subprocess.Popen([sys.executable,'-u','tools/run_cos_restart.py','--run',str(run)],cwd=repo,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                state['runner_pid']=proc.pid;write(stage/'queue_status.json',state);rc=proc.wait()
            result=json.loads((run/'runtime.json').read_text()) if (run/'runtime.json').exists() else {}
            state['runs'].append(dict(run_id=r['run_id'],returncode=rc,phase=result.get('phase','failed'),source_git_commit=r['source_git_commit'],run=str(run)))
            if rc or result.get('phase')!='complete':raise RuntimeError('Stopped at '+r['run_id'])
        state.update(phase='complete',active=None,runner_pid=None,completed_unix=time.time())
    except Exception as e:
        state.update(phase='failed',error=str(e),failed_unix=time.time());raise
    finally:write(stage/'queue_status.json',state)

if __name__=='__main__':main()

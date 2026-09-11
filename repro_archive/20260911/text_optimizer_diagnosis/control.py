"""Commit-pinned deployment, preflight, detached launch and two bounded inspections."""
import argparse
import base64
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo
from analysis import digest,write_json

HERE=Path(__file__).resolve().parent
DOCS=next(p for p in HERE.parents if (p/'.git').exists())
PYTHON='/root/autodl-tmp/envs/betterlvit-paper/bin/python'
KEY='C:/Users/dtftn/.ssh/seetacloud_betterlvit_ed25519'
FILES=('run_screen.py','heads.py','check_heads.py','analysis.py','control.py','manifest.json','PROTOCOL.md')
M=json.loads((HERE/'manifest.json').read_text())
SYDNEY=ZoneInfo('Australia/Sydney')


def remote(code,timeout=180):
    result=subprocess.run(['ssh','-i',KEY,'-p','21465','-o','BatchMode=yes','-o','ConnectTimeout=15',
        'root@connect.westb.seetacloud.com',PYTHON+' -'],input=code,text=True,encoding='utf-8',
        capture_output=True,timeout=timeout)
    if result.returncode:raise RuntimeError(result.stderr+'\n'+result.stdout[-8000:])
    return json.loads(result.stdout)


def frozen_files(sha):
    payload={}
    for name in FILES:
        value=subprocess.check_output(['git','show',sha+':'+(HERE/name).relative_to(DOCS).as_posix()],cwd=DOCS)
        assert value==(HERE/name).read_bytes(),'Committed deployment bytes changed: '+name
        payload[name]=base64.b64encode(value).decode()
    return payload


def deploy():
    assert not (HERE/'deployment.json').exists(),'Deployment already recorded'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=DOCS,text=True).strip(),'Commit sources first'
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=DOCS,text=True).strip()
    root='/root/text_optimizer_d_'+sha[:8]
    result=remote('ROOT='+repr(root)+'\nSHA='+repr(sha)+'\nPAYLOAD='+repr(frozen_files(sha))+'\n'+'''
import base64,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
root=Path(ROOT);root.mkdir(exist_ok=False)
hashes={}
for name,value in PAYLOAD.items():
    assert Path(name).name==name
    data=base64.b64decode(value);(root/name).write_bytes(data)
    hashes[name]=hashlib.sha256(data).hexdigest()
manifest=json.loads((root/'manifest.json').read_text())
assert shutil.disk_usage(root).free>=manifest['cache_budget_bytes']+manifest['minimum_free_after_cache_bytes']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'GPU occupied before new-run preflight'
env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='1219')
tests={}
for device in ('cpu','cuda'):
    r=subprocess.run(['/root/autodl-tmp/envs/betterlvit-paper/bin/python','-B',str(root/'check_heads.py'),'--device',device],
        cwd=root,env=env,text=True,capture_output=True,timeout=70)
    tests[device]=dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr)
    if r.returncode:break
result=dict(source_git_commit=SHA,remote_directory=ROOT,files_sha256=hashes,tests=tests,
    preflight_passed=len(tests)==2 and all(v['exit_code']==0 for v in tests.values()),deployed_unix=time.time(),
    system_free_bytes=shutil.disk_usage(root).free)
(root/'deployment.json').write_text(json.dumps(result))
print(json.dumps(result))
''')
    assert result['files_sha256']=={n:digest(HERE/n) for n in FILES}
    write_json(HERE/'deployment.json',result);print(json.dumps(result))
    assert result['preflight_passed'],'Fix failed preflight before any training'


def launch():
    d=json.loads((HERE/'deployment.json').read_text());assert d['preflight_passed']
    sha=d['source_git_commit'];frozen_files(sha)
    a=HERE.parent/'text_grounding_execution_v3/results/independent_verification.json'
    assert json.loads(a.read_text())['verified']
    b=HERE.parent/'text_head_screening/results/independent_verification.json'
    assert json.loads(b.read_text())['verified']
    assert not (HERE/'launch_attempt.json').exists(),'Reconcile existing receipt before retry'
    write_json(HERE/'launch_attempt.json',dict(source_git_commit=sha,submitted_unix=time.time(),a_proof_sha256=digest(a)))
    result=remote('ROOT='+repr(d['remote_directory'])+'\nSHA='+repr(sha)+'\nHASHES='+repr(d['files_sha256'])+'\n'+'''
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
root=Path(ROOT);manifest=json.loads((root/'manifest.json').read_text())
assert not (root/'launch_receipt.json').exists() and not (root/'results').exists()
for name,sha in HASHES.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'GPU occupied'
assert shutil.disk_usage(root).free>=manifest['cache_budget_bytes']+manifest['minimum_free_after_cache_bytes']
env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='1219')
started=time.time()
with (root/'run.log').open('x') as log:
    p=subprocess.Popen(['/root/autodl-tmp/envs/betterlvit-paper/bin/python','-B','-u',str(root/'run_screen.py'),
        '--output',str(root/'results'),'--source-sha',SHA],cwd=root,env=env,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
result=dict(source_git_commit=SHA,pid=p.pid,submitted_unix=started,remote_directory=ROOT,
    system_free_bytes=shutil.disk_usage(root).free,dispatch_only=True,no_persistent_ssh=True)
(root/'launch_receipt.json').write_text(json.dumps(result));print(json.dumps(result))
''')
    check=math.ceil((result['submitted_unix']+M['first_check_delay_seconds'])/60)*60
    state=dict(source_git_commit=sha,phase='submitted',inspections_completed=0,maximum_inspections=2,
        planned_check_unix=check,planned_check_sydney=datetime.fromtimestamp(check,SYDNEY).isoformat())
    write_json(HERE/'launch.json',result);write_json(HERE/'state.json',state)
    print(json.dumps(dict(launch=result,state=state)))


def collect():
    state=json.loads((HERE/'state.json').read_text());deployment=json.loads((HERE/'deployment.json').read_text())
    assert state['phase'] not in ('complete','failed','missing_runtime') and state['inspections_completed']<2
    assert time.time()>=state['planned_check_unix'],'Honor the predicted appointment'
    state['inspections_completed']+=1;write_json(HERE/'state.json',state)
    result=remote('ROOT='+repr(deployment['remote_directory'])+'\n'+'''
import base64,hashlib,json,time
from pathlib import Path
root=Path(ROOT);path=root/'results/runtime.json'
runtime=json.loads(path.read_text()) if path.exists() else dict(phase='missing_runtime')
out=dict(observed_unix=time.time(),runtime=runtime,files={})
if runtime['phase']=='complete':
    for name,meta in runtime['artifacts'].items():
        assert Path(name).name==name
        data=(root/'results'/name).read_bytes()
        assert len(data)==meta['bytes'] and hashlib.sha256(data).hexdigest()==meta['sha256']
        out['files'][name]=base64.b64encode(data).decode()
out['log_tail']=(root/'run.log').read_text(errors='replace')[-16000:]
print(json.dumps(out))
''')
    files=result.pop('files');write_json(HERE/('inspection_'+str(state['inspections_completed'])+'.json'),result)
    state['phase']=result['runtime']['phase']
    if state['phase']=='complete':
        target=HERE/'results';target.mkdir(exist_ok=False)
        for name,b64 in files.items():(target/name).write_bytes(base64.b64decode(b64))
        write_json(target/'runtime.json',result['runtime'])
        (target/'run.log').write_text(result['log_tail'],encoding='utf-8')
        for event in ('training_completed_unix','completed_unix'):
            delay=result['observed_unix']-result['runtime'][event]
            state[event+'_inspection_delay_seconds']=delay
            state[event+'_check_within_30min']=0<=delay<=1800
    elif state['phase'] not in ('failed','missing_runtime') and state['inspections_completed']<2:
        eta=result['runtime'].get('expected_completion_unix')
        check=math.ceil((max(time.time(),eta)+M['completion_check_margin_seconds'])/60)*60 if eta else math.ceil((time.time()+900)/60)*60
        state.update(expected_completion_unix=eta,planned_check_unix=check,
            planned_check_sydney=datetime.fromtimestamp(check,SYDNEY).isoformat())
    write_json(HERE/'state.json',state)
    if state['phase']=='complete':verify()
    print(json.dumps(dict(state=state,runtime=result['runtime'],files_downloaded=len(files))))


def verify():
    import numpy as np
    target=HERE/'results';runtime=json.loads((target/'runtime.json').read_text())
    assert runtime['phase']=='complete'
    for name,meta in runtime['artifacts'].items():
        assert (target/name).stat().st_size==meta['bytes'] and digest(target/name)==meta['sha256'],name
    summary=json.loads((target/'summary.json').read_text())
    assert summary['source_git_commit']==runtime['source_git_commit']
    assert summary['original_adam_0_256_512_exact_reproduction']
    assert not summary['internal_holdout_evaluated'] and not summary['test_split_accessed']
    history=json.loads((target/'history.json').read_text())
    assert [r['step'] for r in history]==list(range(1,513))
    cases=json.loads((target/'train_diagnostics.json').read_text())[-1]['cases']
    assert len(cases)==18 and len(summary['cases'])==18
    fit=set(json.loads((HERE.parent/'text_head_screening/results/training_order.json').read_text())['fit_indices'])
    for key,value in cases.items():
        rs=value['records'];assert len(rs)==32 and all(r['index'] in fit for r in rs)
        for row in rs:
            for metric in (row['baseline'],row['output']):
                tp,fp,fn=metric['tp'],metric['fp'],metric['fn']
                assert metric['iou']==(tp/(tp+fp+fn) if tp+fp+fn else 0.)
                assert metric['dice']==(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
        assert summary['cases'][key]['delta_iou']==float(np.mean([r['output']['iou']-r['baseline']['iou'] for r in rs]))
        assert summary['cases'][key]['mean_abs_delta']==float(np.mean([r['mean_abs_delta'] for r in rs]))
    proof=dict(verified=True,source_git_commit=runtime['source_git_commit'],cases=18,steps_per_case=512,
        train_diagnostic_samples=32,artifact_hashes_verified=True,summary_recomputed=True,
        original_adam_reproduction=True,internal_holdout_evaluated=False,test_split_accessed=False)
    write_json(target/'independent_verification.json',proof);print(json.dumps(proof))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('deploy','launch','collect','verify'))
    globals()[parser.parse_args().action]()

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
from screen_analysis import summarize

HERE=Path(__file__).resolve().parent
DOCS=next(p for p in HERE.parents if (p/'.git').exists())
PYTHON='/root/autodl-tmp/envs/betterlvit-paper/bin/python'
KEY='C:/Users/dtftn/.ssh/seetacloud_betterlvit_ed25519'
FILES=('run_screen.py','heads.py','heads_names.py','check_heads.py','screen_analysis.py','analysis.py',
    'split_policy.py','text_policy.py','data_identity.py','register.py','control.py','manifest.json','split.json','PROTOCOL.md')
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
    root='/root/text_head_b_'+sha[:8]
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
    target=HERE/'results';runtime=json.loads((target/'runtime.json').read_text())
    assert runtime['phase']=='complete'
    for name,meta in runtime['artifacts'].items():
        assert (target/name).stat().st_size==meta['bytes'] and digest(target/name)==meta['sha256'],name
    data=json.loads((target/'records.json').read_text());rows=data['records']
    split=json.loads((HERE/'split.json').read_text())['records']
    expected=[r for r in split if r['partition']=='holdout']
    assert len(rows)==len(expected)==1131 and sum(r['eligible'] for r in rows)==458
    assert [r['name'] for r in rows]==[r['name'] for r in expected]
    assert data['source_git_commit']==runtime['source_git_commit'] and data['final_step']==M['steps']
    for row,registered in zip(rows,expected):
        for key,value in registered.items():assert row[key]==value
        for value in row['outputs'].values():
            tp,fp,fn=value['tp'],value['fp'],value['fn']
            assert value['iou']==(tp/(tp+fp+fn) if tp+fp+fn else 0.)
            assert value['dice']==(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
        if not row['eligible']:
            assert all(value==row['outputs']['baseline'] for value in row['outputs'].values())
    summary=summarize(rows,data['small_cutoff_fit_gt_pixels'],M['mechanism_screen_gate'])
    assert summary==json.loads((target/'summary.json').read_text()),'Independent summary differs'
    history=json.loads((target/'history.json').read_text())
    assert [r['step'] for r in history]==list(range(1,1025))
    assert abs(history[0]['lr']-M['lr_start'])<1e-15 and abs(history[-1]['lr']-M['lr_end'])<1e-15
    proof=dict(verified=True,source_git_commit=runtime['source_git_commit'],files_sha256_verified=True,
        counts_and_summary_recomputed=True,samples=len(rows),steps_per_head=len(history),
        official_validation_accessed=False,test_split_accessed=False,automatic_full_training=False)
    write_json(target/'independent_verification.json',proof);print(json.dumps(proof))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('deploy','launch','collect','verify'))
    globals()[parser.parse_args().action]()

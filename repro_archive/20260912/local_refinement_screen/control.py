"""Commit-pinned preflight, detached fitting, and at most two predictive checks."""
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
M=json.loads((HERE/'manifest.json').read_text())
FILES=('common.py','refiner.py','mass_projection.py','analysis.py','check_refiner.py','preflight.py','run_train.py','manifest.json','split.json','PROTOCOL.md')
TZ=ZoneInfo('Australia/Sydney')

def remote(code,timeout=180):
    result=subprocess.run(['ssh','-i','C:/Users/dtftn/.ssh/seetacloud_betterlvit_ed25519','-p','21465',
        '-o','BatchMode=yes','-o','ConnectTimeout=15','root@connect.westb.seetacloud.com',PYTHON+' -'],
        input=code,text=True,encoding='utf-8',capture_output=True,timeout=timeout)
    if result.returncode:raise RuntimeError(result.stderr+'\n'+result.stdout[-8000:])
    return json.loads(result.stdout)

def frozen_files(sha):
    payload={}
    for name in FILES:
        raw=subprocess.check_output(['git','show',sha+':'+(HERE/name).relative_to(DOCS).as_posix()],cwd=DOCS)
        assert raw==(HERE/name).read_bytes(),'Deployment source differs: '+name
        payload[name]=base64.b64encode(raw).decode()
    return payload

def preflight_launch():
    assert not (HERE/'deployment_attempt.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=DOCS,text=True).strip()
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=DOCS,text=True).strip();payload=frozen_files(sha)
    root='/root/local_refinement_f_'+sha[:8]
    write_json(HERE/'deployment_attempt.json',dict(source_git_commit=sha,remote_directory=root,submitted_unix=time.time()))
    result=remote('ROOT='+repr(root)+'\nSHA='+repr(sha)+'\nPAYLOAD='+repr(payload)+'\n'+'''
import base64,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
root=Path(ROOT);root.mkdir(exist_ok=False)
hashes={}
for name,value in PAYLOAD.items():
    assert Path(name).name==name
    raw=base64.b64decode(value);(root/name).write_bytes(raw);hashes[name]=hashlib.sha256(raw).hexdigest()
manifest=json.loads((root/'manifest.json').read_text())
free=shutil.disk_usage(root).free
assert free>=manifest['exact_cache_bytes']+manifest['minimum_free_bytes'],free
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'GPU occupied'
started=time.time()
with (root/'preflight.log').open('x') as log:
    p=subprocess.Popen(['/root/autodl-tmp/envs/betterlvit-paper/bin/python','-B','-u',str(root/'preflight.py'),
        '--output',str(root/'preflight'),'--source-sha',SHA],cwd=root,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,
        env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='1219'))
receipt=dict(source_git_commit=SHA,remote_directory=ROOT,pid=p.pid,submitted_unix=started,
    files_sha256=hashes,system_free_before_bytes=free,predicted_collect_unix=started+180)
(root/'deployment.json').write_text(json.dumps(receipt));print(json.dumps(receipt))
''')
    assert result['files_sha256']=={n:digest(HERE/n) for n in FILES}
    write_json(HERE/'deployment.json',result);print(json.dumps(result))

def fetch(root,folder):
    return remote('ROOT='+repr(root)+'\nFOLDER='+repr(folder)+'\n'+'''
import base64,hashlib,json,time
from pathlib import Path
root=Path(ROOT);directory=root/FOLDER
runtime=json.loads((directory/'runtime.json').read_text())
out=dict(runtime=runtime,observed_unix=time.time(),files={})
if runtime['phase']=='complete':
    for name,meta in runtime['artifacts'].items():
        assert Path(name).name==name
        raw=(directory/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==meta['sha256'] and len(raw)==meta['bytes']
        out['files'][name]=base64.b64encode(raw).decode()
out['log']=(root/('preflight.log' if FOLDER=='preflight' else 'train.log')).read_text(errors='replace')
print(json.dumps(out))
''',timeout=240)

def save_result(result,folder):
    raw=result.pop('files');write_json(HERE/(folder+'_collection.json'),result)
    if result['runtime']['phase']=='complete':
        target=HERE/folder;target.mkdir(exist_ok=False)
        for name,value in raw.items():(target/name).write_bytes(base64.b64decode(value))
        write_json(target/'runtime.json',result['runtime'])
        (target/'run.log').write_text(result['log'],encoding='utf-8',newline='\n')
    print(json.dumps(dict(phase=result['runtime']['phase'],runtime=result['runtime'],log_tail=result['log'][-3500:])))

def preflight_collect():
    d=json.loads((HERE/'deployment.json').read_text());assert time.time()>=d['predicted_collect_unix']
    assert not (HERE/'preflight').exists()
    save_result(fetch(d['remote_directory'],'preflight'),'preflight')

def train_launch():
    d=json.loads((HERE/'deployment.json').read_text());sha=d['source_git_commit'];frozen_files(sha)
    preflight=json.loads((HERE/'preflight/runtime.json').read_text());assert preflight['phase']=='complete'
    benchmark=json.loads((HERE/'preflight/benchmark.json').read_text());assert benchmark['passed'] and benchmark['source_git_commit']==sha
    assert not (HERE/'launch_attempt.json').exists()
    write_json(HERE/'launch_attempt.json',dict(source_git_commit=sha,submitted_unix=time.time()))
    result=remote('ROOT='+repr(d['remote_directory'])+'\nSHA='+repr(sha)+'\nHASHES='+repr(d['files_sha256'])+'\n'+'''
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
root=Path(ROOT)
assert not (root/'results').exists() and not (root/'launch.json').exists()
for name,expected in HASHES.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected
assert json.loads((root/'preflight/runtime.json').read_text())['phase']=='complete'
assert shutil.disk_usage(root).free>=256000000
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'GPU occupied'
started=time.time()
with (root/'train.log').open('x') as log:
    p=subprocess.Popen(['/root/autodl-tmp/envs/betterlvit-paper/bin/python','-B','-u',str(root/'run_train.py'),
        '--output',str(root/'results'),'--source-sha',SHA],cwd=root,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,
        env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='1219'))
receipt=dict(source_git_commit=SHA,remote_directory=ROOT,pid=p.pid,submitted_unix=started,
    system_free_bytes=shutil.disk_usage(root).free,no_persistent_ssh=True)
(root/'launch.json').write_text(json.dumps(receipt));print(json.dumps(receipt))
''')
    expected=result['submitted_unix']+benchmark['predicted_train_and_oof_seconds']
    check=math.ceil((expected+180)/60)*60
    state=dict(source_git_commit=sha,phase='submitted',inspections_completed=0,maximum_inspections=2,
        expected_completion_unix=expected,expected_completion_sydney=datetime.fromtimestamp(expected,TZ).isoformat(),
        planned_check_unix=check,planned_check_sydney=datetime.fromtimestamp(check,TZ).isoformat())
    write_json(HERE/'launch.json',result);write_json(HERE/'state.json',state)
    print(json.dumps(dict(launch=result,state=state)))

def train_collect():
    state=json.loads((HERE/'state.json').read_text());d=json.loads((HERE/'deployment.json').read_text())
    assert state['phase'] not in ('complete','failed') and state['inspections_completed']<2
    assert time.time()>=state['planned_check_unix']
    state['inspections_completed']+=1;write_json(HERE/'state.json',state)
    result=fetch(d['remote_directory'],'results');runtime=result['runtime'];observed=result['observed_unix']
    if runtime['phase']=='complete':
        for event in ('training_completed_unix','completed_unix'):
            state[event+'_inspection_delay_seconds']=observed-runtime[event]
            state[event+'_within30min']=0<=observed-runtime[event]<=1800
    elif runtime['phase']!='failed' and state['inspections_completed']<2:
        expected=runtime.get('expected_completion_unix',observed+600)
        check=math.ceil((max(expected,observed+60)+180)/60)*60
        state.update(planned_check_unix=check,planned_check_sydney=datetime.fromtimestamp(check,TZ).isoformat())
    state['phase']=runtime['phase'];write_json(HERE/'state.json',state)
    save_result(result,'results')
    print(json.dumps(state))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('preflight_launch','preflight_collect','train_launch','train_collect'));a=ap.parse_args()
    globals()[a.action]()

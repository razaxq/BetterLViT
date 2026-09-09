"""Deploy frozen sources, short preflight, detached launch, predicted final collection."""
import argparse
import base64
import json
import math
from pathlib import Path
import subprocess
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from analysis import digest, summarize, write_json

HERE=Path(__file__).resolve().parent
DOCS=HERE.parents[2]
OUT=Path('D:/BetterLViT/outputs/semantic_probe_20260910')
REMOTE='/root/semantic_probe_20260910'
PYTHON='/root/autodl-tmp/envs/betterlvit-paper/bin/python'
KEY='C:/Users/dtftn/.ssh/seetacloud_betterlvit_ed25519'
FILES=('run_probe.py','analysis.py','check_analysis.py','manifest.json','PLAN.md','r2_validation.json')


def remote(code):
    r=subprocess.run(['ssh','-i',KEY,'-p','21465','-o','BatchMode=yes','-o','ConnectTimeout=15',
        'root@connect.westb.seetacloud.com',PYTHON+' -'],input=code,text=True,encoding='utf-8',capture_output=True,timeout=60)
    if r.returncode: raise RuntimeError(r.stderr+'\n'+r.stdout[-6000:])
    return json.loads(r.stdout)


def source():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=DOCS,text=True).strip(),'Commit all sources first'
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=DOCS,text=True).strip()
    payload={}
    for name in FILES:
        blob=subprocess.check_output(['git','show',sha+':'+(HERE/name).relative_to(DOCS).as_posix()],cwd=DOCS)
        assert blob==(HERE/name).read_bytes()
        payload[name]=base64.b64encode(blob).decode()
    return sha,payload


def preflight():
    sha,payload=source(); OUT.mkdir(exist_ok=True)
    result=remote('ROOT='+repr(REMOTE)+'\nSHA='+repr(sha)+'\nPAYLOAD='+repr(payload)+'\n'+'''
import base64,json,os,shutil,subprocess,time
from pathlib import Path
root=Path(ROOT)
assert not (root/'launch_receipt.json').exists(), 'Already launched'
root.mkdir(exist_ok=True)
for name,data in PAYLOAD.items():
    assert Path(name).name==name
    (root/name).write_bytes(base64.b64decode(data))
assert shutil.disk_usage('/root').free>1_000_000_000
shared=int(subprocess.check_output(['du','-sb','/autodl-fs/data'],text=True).split()[0]);assert shared<20_000_000_000
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
assert not apps, 'GPU is occupied: '+apps
env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='1219')
with (root/'preflight.log').open('w') as log:
    job=subprocess.run(['/root/autodl-tmp/envs/betterlvit-paper/bin/python','-u',str(root/'run_probe.py'),
        '--preflight','--output',str(root/'preflight.json'),'--source-sha',SHA],cwd=root,env=env,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=48)
if job.returncode:raise RuntimeError((root/'preflight.log').read_text()[-6000:])
r=json.loads((root/'preflight.json').read_text());r.update(diagnostic_source_git_commit=SHA,shared_fs_bytes=shared)
(root/'preflight_receipt.json').write_text(json.dumps(r))
print(json.dumps(r))
''')
    assert result['status']=='ok' and result['diagnostic_source_git_commit']==sha
    for name in ('run_probe.py','analysis.py','check_analysis.py'):assert result['scripts_sha256'][name]==digest(HERE/name)
    write_json(OUT/'preflight.json',result); print(json.dumps(result))


def launch():
    sha,_=source()
    pf=json.loads((OUT/'preflight.json').read_text())
    assert pf['diagnostic_source_git_commit']==sha
    assert not (OUT/'launch_attempt.json').exists(),'Reconcile receipt before any retry'
    write_json(OUT/'launch_attempt.json',dict(submitted_unix=time.time(),source_git_commit=sha))
    hashes={n:digest(HERE/n) for n in FILES}
    result=remote('ROOT='+repr(REMOTE)+'\nSHA='+repr(sha)+'\nHASHES='+repr(hashes)+'\n'+'''
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
root=Path(ROOT)
assert not (root/'launch_receipt.json').exists() and not (root/'results').exists()
for n,h in HASHES.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h
pf=json.loads((root/'preflight_receipt.json').read_text());assert pf['diagnostic_source_git_commit']==SHA and pf['status']=='ok'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='1219')
started=time.time()
with (root/'run.log').open('x') as log:
    p=subprocess.Popen(['/root/autodl-tmp/envs/betterlvit-paper/bin/python','-u',str(root/'run_probe.py'),
        '--output',str(root/'results'),'--source-sha',SHA],cwd=root,env=env,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
receipt=dict(pid=p.pid,submitted_unix=started,diagnostic_source_git_commit=SHA,deployed_sha256=HASHES,
    system_free_bytes=shutil.disk_usage('/root').free,remote_directory=str(root))
(root/'launch_receipt.json').write_text(json.dumps(receipt))
print(json.dumps(receipt))
''')
    expected=result['submitted_unix']+pf['predicted_run_seconds']
    check=math.ceil((expected+300)/60)*60
    state=dict(phase='submitted',diagnostic_source_git_commit=sha,inspections_completed=0,maximum_inspections=2,
        expected_completion_unix=expected,planned_check_unix=check,no_persistent_ssh=True,
        expected_completion_sydney=datetime.fromtimestamp(expected,ZoneInfo('Australia/Sydney')).isoformat(),
        planned_check_sydney=datetime.fromtimestamp(check,ZoneInfo('Australia/Sydney')).isoformat())
    write_json(OUT/'launch.json',result);write_json(OUT/'state.json',state)
    print(json.dumps(dict(launch=result,state=state)))


def collect():
    state=json.loads((OUT/'state.json').read_text())
    assert state['phase'] not in ('complete','failed') and state['inspections_completed']<2
    assert time.time()>=state['planned_check_unix'],'Too early; honor predicted check'
    state['inspections_completed']+=1;write_json(OUT/'state.json',state)
    result=remote('ROOT='+repr(REMOTE)+'\n'+'''
import base64,hashlib,json,time
from pathlib import Path
root=Path(ROOT);p=root/'results'/'runtime.json'
r=json.loads(p.read_text()) if p.exists() else dict(phase='missing_runtime')
out=dict(observed_unix=time.time(),runtime=r,files={})
if r['phase']=='complete':
    for name,meta in r['artifacts'].items():
        data=(root/'results'/name).read_bytes();assert len(data)==meta['bytes'] and hashlib.sha256(data).hexdigest()==meta['sha256']
        out['files'][name]=base64.b64encode(data).decode()
if r['phase'] in ('complete','failed','missing_runtime'):
    out['log_tail']=(root/'run.log').read_text(errors='replace')[-8000:]
print(json.dumps(out))
''')
    files=result.pop('files');write_json(OUT/('inspection_'+str(state['inspections_completed'])+'.json'),result)
    state['phase']=result['runtime']['phase']
    if state['phase']=='complete':
        target=OUT/'results';target.mkdir(exist_ok=False)
        for name,b64 in files.items():
            assert Path(name).name==name
            data=base64.b64decode(b64);path=target/name;path.write_bytes(data)
            assert digest(path)==result['runtime']['artifacts'][name]['sha256']
        write_json(target/'runtime.json',result['runtime'])
        (target/'run.log').write_text(result['log_tail'],encoding='utf-8')
        records=json.loads((target/'records.json').read_text())
        summary=json.loads((target/'summary.json').read_text())
        assert records['diagnostic_source_git_commit']==state['diagnostic_source_git_commit']
        assert not records['test_split_accessed'] and records['baseline_state_unchanged']
        assert len(records['records'])==1429 and records['baseline_per_image_max_difference']<1e-12
        assert summarize(records['records'])==summary,'Independent CPU summary mismatch'
        state['seconds_after_completion']=result['observed_unix']-result['runtime']['completed_unix']
        state['completion_check_within_30min']=0<=state['seconds_after_completion']<=1800
        state['proceed_to_architecture']=summary['proceed_to_architecture']
    write_json(OUT/'state.json',state)
    print(json.dumps(dict(state=state,runtime=result['runtime'],files_downloaded=len(files))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('preflight','launch','collect'));a=p.parse_args()
    globals()[a.action]()

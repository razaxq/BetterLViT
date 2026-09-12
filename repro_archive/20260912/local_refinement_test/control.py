"""Commit-pinned detached Test launch and bounded, predicted artifact collection."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zlib
from analysis import digest,write_json
HERE=Path(__file__).resolve().parent;DOCS=HERE.parents[2]
PY='/root/autodl-tmp/envs/betterlvit-paper/bin/python'
FILES=('evaluate.py','authorization.json','PROTOCOL.md','refiner.py','mass_projection.py','analysis.py','screen_analysis.py','refiner_names.py')
def remote(code,timeout=120):
    p=subprocess.run(['ssh','-i','C:/Users/dtftn/.ssh/seetacloud_betterlvit_ed25519','-p','21465','-o','BatchMode=yes',
        '-o','ConnectTimeout=15','root@connect.westb.seetacloud.com',PY+' -'],input=code,text=True,encoding='utf-8',
        capture_output=True,timeout=timeout)
    if p.returncode:raise RuntimeError(p.stderr+'\n'+p.stdout[-4000:])
    return json.loads(p.stdout)
def launch():
    assert not (HERE/'launch_attempt.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=DOCS,text=True).strip()
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=DOCS,text=True).strip()
    payload={}
    for name in FILES:
        raw=subprocess.check_output(['git','show',sha+':'+(HERE/name).relative_to(DOCS).as_posix()],cwd=DOCS)
        assert raw==(HERE/name).read_bytes(),name
        payload[name]=base64.b64encode(zlib.compress(raw)).decode()
    a=json.loads((HERE/'authorization.json').read_text(encoding='utf-8'))
    assert set(a['inherited_files_sha256'])<=set(FILES),'Every verified dependency must be deployed'
    historical=DOCS/a['historical_test_relative'];assert digest(historical)==a['historical_test_sha256']
    payload['r2_historical_test.json']=base64.b64encode(zlib.compress(historical.read_bytes())).decode()
    root='/root/autodl-tmp/local_refinement_test_'+sha[:8]
    write_json(HERE/'launch_attempt.json',dict(evaluation_source_git_commit=sha,remote_directory=root,attempt_unix=time.time()))
    result=remote('ROOT='+repr(root)+'\nSHA='+repr(sha)+'\nPAYLOAD='+repr(payload)+'\n'+'''
import base64,hashlib,json,os,shutil,subprocess,time,zlib
from pathlib import Path
root=Path(ROOT);root.mkdir(exist_ok=False)
assert shutil.disk_usage(root).free>=256000000
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'GPU occupied'
hashes={}
for name,value in PAYLOAD.items():
    assert Path(name).name==name
    raw=zlib.decompress(base64.b64decode(value));(root/name).write_bytes(raw);hashes[name]=hashlib.sha256(raw).hexdigest()
authorization=json.loads((root/'authorization.json').read_text())
assert set(authorization['inherited_files_sha256'])<=set(hashes)
env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='1219',PYTHONDONTWRITEBYTECODE='1')
with (root/'preflight.log').open('x') as log:
    check=subprocess.run(['/root/autodl-tmp/envs/betterlvit-paper/bin/python','-B','-u',str(root/'evaluate.py'),
        '--output',str(root/'preflight'),'--source-sha',SHA,'--preflight-only'],cwd=root,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,env=env,timeout=90)
assert check.returncode==0,(root/'preflight.log').read_text()[-4000:]
preflight=json.loads((root/'preflight/preflight.json').read_text());assert preflight['verified'] and preflight['test_images_loaded']==0
started=time.time()
with (root/'evaluate.log').open('x') as log:
    p=subprocess.Popen(['/root/autodl-tmp/envs/betterlvit-paper/bin/python','-B','-u',str(root/'evaluate.py'),
        '--output',str(root/'results'),'--source-sha',SHA],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
        start_new_session=True,close_fds=True,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='1219',PYTHONDONTWRITEBYTECODE='1'))
receipt=dict(evaluation_source_git_commit=SHA,remote_directory=ROOT,pid=p.pid,submitted_unix=started,
    deployed_sha256=hashes,first_collect_unix=started+360,no_persistent_ssh=True,free_bytes=shutil.disk_usage(root).free,
    model_and_dependency_preflight=preflight)
(root/'launch.json').write_text(json.dumps(receipt));print(json.dumps(receipt))
''')
    assert all(result['deployed_sha256'][n]==digest(HERE/n) for n in FILES)
    write_json(HERE/'launch.json',result)
    write_json(HERE/'state.json',dict(phase='submitted',inspections=1,maximum_inspections=2,planned_collect_unix=result['first_collect_unix'],
        first_inspection_failed_before_test_images=True))
    print(json.dumps(result))
def collect():
    d=json.loads((HERE/'launch.json').read_text());state=json.loads((HERE/'state.json').read_text())
    assert state['phase'] not in ('complete','failed') and state['inspections']<2
    assert time.time()>=state['planned_collect_unix']
    state['inspections']+=1;write_json(HERE/'state.json',state)
    result=remote('ROOT='+repr(d['remote_directory'])+'\n'+'''
import base64,hashlib,json,time,zlib
from pathlib import Path
root=Path(ROOT);runtime=json.loads((root/'results/runtime.json').read_text())
result=dict(runtime=runtime,observed_unix=time.time(),files={},log=(root/'evaluate.log').read_text(errors='replace'))
if runtime['phase']=='complete':
    for name,meta in runtime['artifacts'].items():
        raw=(root/'results'/name).read_bytes();assert len(raw)==meta['bytes'] and hashlib.sha256(raw).hexdigest()==meta['sha256']
        result['files'][name]=base64.b64encode(zlib.compress(raw)).decode()
print(json.dumps(result))
''',timeout=180)
    state['phase']=result['runtime']['phase']
    if state['phase']=='complete':
        directory=HERE/'results';directory.mkdir(exist_ok=False)
        for name,value in result.pop('files').items():
            assert Path(name).name==name
            raw=zlib.decompress(base64.b64decode(value));meta=result['runtime']['artifacts'][name]
            assert len(raw)==meta['bytes'] and hashlib.sha256(raw).hexdigest()==meta['sha256']
            (directory/name).write_bytes(raw)
        state['completion_delay_seconds']=result['observed_unix']-result['runtime']['completed_unix']
        state['completion_observed_within30min']=0<=state['completion_delay_seconds']<=1800
        write_json(directory/'runtime.json',result['runtime'])
        (directory/'run.log').write_text(result['log'],encoding='utf-8',newline='\n')
    elif state['phase']!='failed':
        state['planned_collect_unix']=max(result['observed_unix']+120,result['runtime'].get('expected_completion_unix',0)+120)
    write_json(HERE/'collection.json',result);write_json(HERE/'state.json',state)
    print(json.dumps(dict(state=state,runtime=result['runtime'],log_tail=result['log'][-3000:])))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('launch','collect'));a=ap.parse_args();globals()[a.action]()

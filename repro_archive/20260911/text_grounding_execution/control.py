"""Committed deployment, dependency-gated detached launch and at most two inspections."""
import argparse
import base64
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo
from analysis import digest,summarize,write_json

HERE=Path(__file__).resolve().parent
DOCS=HERE.parents[2]
REMOTE='/root/text_grounding_a_20260911'
PYTHON='/root/autodl-tmp/envs/betterlvit-paper/bin/python'
KEY='C:/Users/dtftn/.ssh/seetacloud_betterlvit_ed25519'
FILES=('run_diagnostic.py','analysis.py','text_policy.py','routing.py','check_policy.py',
       'check_corpus.py','control.py','manifest.json','PLAN.md','r2_validation.json')
MANIFEST=json.loads((HERE/'manifest.json').read_text())
SYDNEY=ZoneInfo('Australia/Sydney')  # Fail before any launch if local tzdata is unavailable.


def remote(code):
    r=subprocess.run(['ssh','-i',KEY,'-p','21465','-o','BatchMode=yes','-o','ConnectTimeout=15',
        'root@connect.westb.seetacloud.com',PYTHON+' -'],input=code,text=True,encoding='utf-8',
        capture_output=True,timeout=60)
    if r.returncode:raise RuntimeError(r.stderr+'\n'+r.stdout[-6000:])
    return json.loads(r.stdout)


def committed_files(sha):
    assert len(sha)==40 and all(c in '0123456789abcdef' for c in sha)
    payload={}
    for name in FILES:
        blob=subprocess.check_output(['git','show',sha+':'+(HERE/name).relative_to(DOCS).as_posix()],cwd=DOCS)
        assert blob==(HERE/name).read_bytes(), 'Frozen deployment file changed: '+name
        payload[name]=base64.b64encode(blob).decode()
    return payload


def deploy():
    assert not (HERE/'deployment.json').exists(),'Deployment already recorded; do not replace its frozen source'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=DOCS,text=True).strip(),'Commit sources first'
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=DOCS,text=True).strip()
    payload=committed_files(sha)
    result=remote('ROOT='+repr(REMOTE)+'\nSHA='+repr(sha)+'\nPAYLOAD='+repr(payload)+'\n'+'''
import base64,hashlib,json,os,subprocess,time
from pathlib import Path
root=Path(ROOT);root.mkdir(exist_ok=True)
assert not (root/'launch_receipt.json').exists() and not (root/'results').exists()
hashes={}
for name,value in PAYLOAD.items():
    assert Path(name).name==name
    data=base64.b64decode(value);path=root/name
    if path.exists():assert path.read_bytes()==data,'Existing deployment differs'
    else:path.write_bytes(data)
    hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONHASHSEED='1219')
r=subprocess.run(['/root/autodl-tmp/envs/betterlvit-paper/bin/python','-B',str(root/'check_policy.py')],
    cwd=root,env=env,capture_output=True,text=True,timeout=30)
assert r.returncode==0,r.stdout+r.stderr
result=dict(diagnostic_source_git_commit=SHA,files_sha256=hashes,remote_directory=ROOT,
    deployed_unix=time.time(),cpu_tests_passed=True,cpu_test_output=r.stdout+r.stderr,
    gpu_queried=False,training_status_queried=False)
(root/'deployment.json').write_text(json.dumps(result))
print(json.dumps(result))
''')
    assert result['diagnostic_source_git_commit']==sha and result['cpu_tests_passed']
    assert result['files_sha256']=={n:digest(HERE/n) for n in FILES}
    write_json(HERE/'deployment.json',result)
    print(json.dumps(result))


def dependency_proofs():
    folder=HERE.parent/'regional_supervision_execution'/'rs3_results'
    expected='f79842331e4e5e41526ed66da31ec174ea4a61b2'
    proofs={}
    for name in ('independent_verification.json','hf_upload_verified.json','download_xet_verified.json'):
        path=folder/name;value=json.loads(path.read_text())
        assert value['verified'] and value['source_git_commit']==expected,(name,'incomplete or wrong source')
        proofs[name]=digest(path)
    value=json.loads((folder/'runtime.json').read_text())
    assert value['phase']=='complete' and value['source_git_commit']==expected
    proofs['runtime.json']=digest(folder/'runtime.json')
    return proofs


def launch():
    assert time.time()>=MANIFEST['minimum_launch_unix'],'Wait for the existing 22:00 RS3 appointment'
    proofs=dependency_proofs()
    deployment=json.loads((HERE/'deployment.json').read_text())
    sha=deployment['diagnostic_source_git_commit'];committed_files(sha)
    assert not (HERE/'launch_attempt.json').exists(),'Reconcile the existing receipt before retrying'
    write_json(HERE/'launch_attempt.json',dict(submitted_unix=time.time(),diagnostic_source_git_commit=sha,rs3_proofs=proofs))
    result=remote('ROOT='+repr(REMOTE)+'\nSHA='+repr(sha)+'\nHASHES='+repr(deployment['files_sha256'])+'\n'+'''
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
root=Path(ROOT);manifest=json.loads((root/'manifest.json').read_text())
assert time.time()>=manifest['minimum_launch_unix']
assert not (root/'launch_receipt.json').exists() and not (root/'results').exists()
for name,h in HASHES.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h
rs3=json.loads((Path(manifest['rs3_dependency_run'])/'runtime.json').read_text())
assert rs3['phase']=='complete' and rs3['source_git_commit']=='f79842331e4e5e41526ed66da31ec174ea4a61b2'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'GPU is occupied'
assert shutil.disk_usage(root).free>100_000_000
env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='1219')
started=time.time()
with (root/'run.log').open('x') as log:
    p=subprocess.Popen(['/root/autodl-tmp/envs/betterlvit-paper/bin/python','-B','-u',str(root/'run_diagnostic.py'),
        '--output',str(root/'results'),'--source-sha',SHA],cwd=root,env=env,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
result=dict(pid=p.pid,submitted_unix=started,diagnostic_source_git_commit=SHA,deployed_sha256=HASHES,
    system_free_bytes=shutil.disk_usage(root).free,remote_directory=str(root),dispatch_only=True)
(root/'launch_receipt.json').write_text(json.dumps(result))
print(json.dumps(result))
''')
    check=math.ceil((result['submitted_unix']+MANIFEST['first_check_delay_seconds'])/60)*60
    state=dict(phase='submitted',diagnostic_source_git_commit=sha,inspections_completed=0,maximum_inspections=2,
        planned_check_unix=check,planned_check_sydney=datetime.fromtimestamp(check,SYDNEY).isoformat(),
        no_persistent_ssh=True,rs3_dependency_proofs=proofs)
    write_json(HERE/'launch.json',result);write_json(HERE/'state.json',state)
    print(json.dumps(dict(launch=result,state=state)))


def collect():
    state=json.loads((HERE/'state.json').read_text())
    assert state['phase'] not in ('complete','failed','missing_runtime') and state['inspections_completed']<2
    assert time.time()>=state['planned_check_unix'],'Too early; honor the appointment'
    state['inspections_completed']+=1;write_json(HERE/'state.json',state)
    result=remote('ROOT='+repr(REMOTE)+'\n'+'''
import base64,hashlib,json,time
from pathlib import Path
root=Path(ROOT);path=root/'results'/'runtime.json'
runtime=json.loads(path.read_text()) if path.exists() else dict(phase='missing_runtime')
out=dict(observed_unix=time.time(),runtime=runtime,files={})
if runtime['phase']=='complete':
    for name,meta in runtime['artifacts'].items():
        assert Path(name).name==name
        data=(root/'results'/name).read_bytes()
        assert len(data)==meta['bytes'] and hashlib.sha256(data).hexdigest()==meta['sha256']
        out['files'][name]=base64.b64encode(data).decode()
out['log_tail']=(root/'run.log').read_text(errors='replace')[-8000:]
print(json.dumps(out))
''')
    files=result.pop('files');number=state['inspections_completed']
    write_json(HERE/('inspection_'+str(number)+'.json'),result)
    state['phase']=result['runtime']['phase']
    write_json(HERE/'state.json',state)
    if state['phase']=='complete':
        target=HERE/'results';target.mkdir(exist_ok=False)
        for name,b64 in files.items():
            assert Path(name).name==name
            path=target/name;path.write_bytes(base64.b64decode(b64))
            assert digest(path)==result['runtime']['artifacts'][name]['sha256']
        write_json(target/'runtime.json',result['runtime'])
        (target/'run.log').write_text(result['log_tail'],encoding='utf-8')
        state['seconds_after_completion']=result['observed_unix']-result['runtime']['completed_unix']
        state['completion_check_within_30min']=0<=state['seconds_after_completion']<=1800
        write_json(HERE/'state.json',state)
        verify()
    elif state['phase'] not in ('failed','missing_runtime') and number<2:
        eta=result['runtime'].get('expected_completion_unix')
        # A preflight with no ETA is bounded to ten minutes in the runner.
        check=math.ceil((max(time.time(),eta)+MANIFEST['completion_check_margin_seconds'])/60)*60 if eta else math.ceil((time.time()+900)/60)*60
        state.update(expected_completion_unix=eta,planned_check_unix=check,
            planned_check_sydney=datetime.fromtimestamp(check,SYDNEY).isoformat())
    write_json(HERE/'state.json',state)
    print(json.dumps(dict(state=state,runtime=result['runtime'],files_downloaded=len(files))))


def verify():
    """Static local reanalysis; never contacts the server or spends an inspection."""
    target=HERE/'results';state=json.loads((HERE/'state.json').read_text())
    runtime=json.loads((target/'runtime.json').read_text())
    for name,meta in runtime['artifacts'].items():
        assert (target/name).stat().st_size==meta['bytes'] and digest(target/name)==meta['sha256']
    record=json.loads((target/'records.json').read_text());summary=json.loads((target/'summary.json').read_text())
    rows=record['records'];assert len(rows)==len({r['name'] for r in rows})==1429
    assert record['diagnostic_source_git_commit']==state['diagnostic_source_git_commit']
    assert record['baseline_source_git_commit']==MANIFEST['baseline_source_git_commit']
    assert record['baseline_checkpoint_sha256']==MANIFEST['baseline_checkpoint_sha256']
    assert not record['test_split_accessed'] and not record['model_updated'] and record['baseline_state_unchanged']
    assert record['baseline_per_image_max_difference']<1e-12
    for row in rows:
        for v in [row['baseline'],*row['conditions'].values()]:
            tp,fp,fn=v['tp'],v['fp'],v['fn']
            assert v['iou']==(tp/(tp+fp+fn) if tp+fp+fn else 0.)
            assert v['dice']==(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    assert summarize(rows)==summary,'Independent CPU reanalysis differs'
    proof=dict(verified=True,diagnostic_source_git_commit=state['diagnostic_source_git_commit'],
        samples=len(rows),test_split_accessed=False,model_updated=False,counts_and_summaries_recomputed=True,
        artifact_hashes_verified=True,automatic_architecture_pass=False)
    write_json(target/'independent_verification.json',proof)
    print(json.dumps(proof))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('deploy','launch','collect','verify'))
    globals()[p.parse_args().action]()

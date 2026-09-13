"""Bounded SSH dispatch/collection for an inference-only job."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = HERE.parents[2]
KEY = 'C:/Users/dtftn/.ssh/seetacloud_betterlvit_ed25519'
HOST = 'root@connect.westb.seetacloud.com'
PYTHON = '/root/autodl-tmp/envs/betterlvit-paper/bin/python'
REMOTE = '/root/autodl-tmp/rapid_iou_20260913'


def save(name, data):
    p = HERE/name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8', newline='\n')


def remote(code):
    p = subprocess.run(['ssh','-i',KEY,'-p','21465','-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,PYTHON+' -'],
        input=code, text=True, encoding='utf-8', capture_output=True, timeout=120)
    if p.returncode:
        raise RuntimeError(p.stderr+p.stdout[-3000:])
    return json.loads(p.stdout)


def copy(path, destination):
    subprocess.run(['scp','-i',KEY,'-P','21465','-o','BatchMode=yes',str(path),HOST+':'+destination],
                   capture_output=True,check=True,timeout=60)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('audit','deploy','launch','collect'))
    p.add_argument('--split', choices=('validation','test'), default='validation')
    a = p.parse_args()
    plan = json.loads((HERE/'plan.json').read_text())
    if a.action == 'audit':
        result = remote('SOURCE='+repr(plan['sources'][0])+'\n'+'''
import json,shutil,subprocess
from pathlib import Path
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
assert not gpu,'GPU is occupied; do not compete with active work'
s=SOURCE
assert Path(s['checkpoint']).is_file()
assert subprocess.check_output(['git','-C',s['repository'],'rev-parse','HEAD'],text=True).strip()==s['source_git_commit']
assert not subprocess.check_output(['git','-C',s['repository'],'status','--porcelain','--untracked-files=no'],text=True).strip()
fs=int(subprocess.check_output(['du','-sb','/autodl-fs/data'],text=True).split()[0]);assert fs<20_000_000_000
free=shutil.disk_usage('/root/autodl-tmp').free;assert free>1_000_000_000
print(json.dumps(dict(verified=True,gpu_idle=True,scratch_free_bytes=free,shared_bytes=fs,checkpoint_exists=True)))
''')
        save('resource_audit.json',result)
    elif a.action == 'deploy':
        sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=DOCS,text=True).strip()
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=DOCS,text=True).strip()
        files=['evaluate.py','plan.json','PROTOCOL.md','runner.py','references/r2s1219_validation.json']
        deployment=dict(evaluation_source_git_commit=sha,sha256={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in files})
        result=remote('ROOT='+repr(REMOTE)+'\n'+'''
import json
from pathlib import Path
p=Path(ROOT);assert not (p/'deployment.json').exists(),'Never replace an existing deployment'
(p/'references').mkdir(parents=True,exist_ok=True)
print(json.dumps(dict(directory_created=True)))
''')
        save('deployment.json',deployment)
        for name in files+['deployment.json']:
            copy(HERE/name,REMOTE+'/'+name)
        result=deployment
    elif a.action == 'launch':
        assert not (HERE/(a.split+'_launch.json')).exists(), 'Already dispatched'
        if a.split=='test':
            selection=json.loads((HERE/'selection.json').read_text());assert selection['methods'] and selection['frozen']
            copy(HERE/'selection.json',REMOTE+'/selection.json')
            ref=DOCS/'repro_archive/20260910/recipe_test/results/r2s1219_test.json'
            copy(ref,REMOTE+'/references/r2s1219_test.json')
        result=remote('ROOT='+repr(REMOTE)+'\nSPLIT='+repr(a.split)+'\nPYTHON='+repr(PYTHON)+'\n'+'''
import json,os,subprocess,time
from pathlib import Path
root=Path(ROOT);run=root/SPLIT;run.mkdir(exist_ok=True)
assert not (run/'runtime.json').exists() and not (run/'runner.log').exists()
with (run/'runner.log').open('w') as log:
    proc=subprocess.Popen([PYTHON,str(root/'runner.py'),SPLIT],stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
print(json.dumps(dict(pid=proc.pid,split=SPLIT,started_unix=time.time(),detached=True)))
''')
        if a.split=='test':
            v=json.loads((HERE/'validation/result.json').read_text())
            runtime=json.loads((HERE/'validation/runtime.json').read_text())
            setup=max(0,runtime['ended_unix']-runtime['started_unix']-v['inference_seconds'])
            seconds=setup+v['inference_seconds']*plan['test_samples']/plan['validation_samples']+45
            result['prediction_method']='completed_validation_setup_plus_sample_scaled_inference_plus_45_seconds'
        else:
            seconds=240
        result['predicted_check_unix']=result['started_unix']+seconds
        save(a.split+'_launch.json',result)
    else:
        result=remote('ROOT='+repr(REMOTE)+'\nSPLIT='+repr(a.split)+'\n'+'''
import hashlib,json
from pathlib import Path
root=Path(ROOT)/SPLIT
runtime=json.loads((root/'runtime.json').read_text())
result=dict(runtime=runtime)
if runtime['phase'] in ('complete','failed'):
    result['files']={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in root.iterdir() if p.is_file()}
    result['log_tail']=(root/'evaluation.log').read_text(errors='replace')[-2000:]
print(json.dumps(result))
''')
        save(a.split+'_collection.json',result)
        if result['runtime']['phase']=='complete':
            folder=HERE/a.split;folder.mkdir(exist_ok=True)
            for name, info in result['files'].items():
                subprocess.run(['scp','-i',KEY,'-P','21465','-o','BatchMode=yes',HOST+':'+REMOTE+'/'+a.split+'/'+name,str(folder/name)],capture_output=True,check=True,timeout=60)
                data=(folder/name).read_bytes();assert len(data)==info['bytes'] and hashlib.sha256(data).hexdigest()==info['sha256']
            save(a.split+'/download_verified.json',dict(verified=True,files=result['files']))
    print(json.dumps(result))


if __name__=='__main__':main()

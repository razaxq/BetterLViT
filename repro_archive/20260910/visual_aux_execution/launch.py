"""Idempotently dispatch one frozen run and disconnect; no health polling."""
import argparse
from datetime import datetime
import json
import math
from zoneinfo import ZoneInfo
from remote_ops import HERE,remote,save
from verify_preflight import verify


def main():
    p=argparse.ArgumentParser();p.add_argument('--label',choices=('s1','s2'),required=True);label=p.parse_args().label
    verify();source=json.loads((HERE/'sources.json').read_text())[label]
    assert not (HERE/(label+'_state.json')).exists(),'Already launched'
    if label=='s2':
        previous_path=HERE/'s1_final_snapshot.json'
        if not previous_path.exists():previous_path=HERE/'s1_first_snapshot.json'
        previous=json.loads(previous_path.read_text())
        assert previous['runtime']['phase']=='complete'
        assert (HERE/'s1_results/README.md').exists(),'Archive and record S1 before launching S2'
        audit=json.loads((HERE/'s1_results/independent_verification.json').read_text())
        backup=json.loads((HERE/'s1_results/hf_upload_verified.json').read_text())
        assert audit['verified'] and backup['verified'] and backup['independent_local_listing_verified']
        assert backup['original_models_preserved']
        assert audit['source_git_commit']==backup['source_git_commit']==previous['runtime']['source_git_commit']
    result=remote('SOURCE='+repr(source)+'\n'+'''
import json,os,shutil,subprocess,time
from pathlib import Path
repo=Path(SOURCE['repository']);run=Path(SOURCE['remote_run'])
receipt=run.parent/(run.name+'_dispatch.json')
if receipt.exists():
    value=json.loads(receipt.read_text());assert value['source_git_commit']==SOURCE['source_git_commit']
else:
    assert not run.exists()
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip();assert sha==SOURCE['source_git_commit']
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
    assert subprocess.run(['pgrep','-f','[t]rain_model.py'],capture_output=True).returncode==1
    free=shutil.disk_usage(repo).free;assert free>4_000_000_000
    fs=int(subprocess.check_output(['du','-sb','/autodl-fs/data'],text=True).split()[0]);assert fs<20_000_000_000
    with (run.parent/(run.name+'_launcher.log')).open('x') as log:
        child=subprocess.Popen(['/root/autodl-tmp/envs/betterlvit-paper/bin/python','-u','tools/run_visual_aux_experiment.py','--run',str(run)],
            cwd=repo,env=dict(os.environ),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    value=dict(source_git_commit=sha,repository=str(repo),remote_run=str(run),pid=child.pid,
        started_unix=time.time(),dispatch_only=True,inspections_completed=0,
        scratch_free_bytes=free,fs_bytes=fs,test_split_accessed=False)
    receipt.write_text(json.dumps(value,indent=2)+'\\n')
print(json.dumps(value))
''')
    result['planned_first_check_unix']=math.ceil((result['started_unix']+900)/60)*60
    for key in ('started_unix','planned_first_check_unix'):
        result[key.replace('_unix','_sydney')]=datetime.fromtimestamp(result[key],ZoneInfo('Australia/Sydney')).isoformat()
    result['experiment_tag']=source['experiment_tag']
    save(label+'_launch.json',result);save(label+'_state.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

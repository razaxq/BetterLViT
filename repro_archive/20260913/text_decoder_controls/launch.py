"""Dispatch one frozen control and disconnect; no training status polling."""
import argparse
from datetime import datetime
import json
import math
from zoneinfo import ZoneInfo
from remote_ops import HERE,DOCS,read,remote,save
p=argparse.ArgumentParser();p.add_argument('--label',choices=('t1','t2'),required=True);label=p.parse_args().label
source=read(HERE/'sources.json')[label]
proof=read(HERE/'preflight_verified.json');assert proof['verified']
assert proof['candidates'][label]['source_git_commit']==source['source_git_commit']
github=read(HERE/'github_sources_verified.json');assert github['verified']
assert github['refs']['refs/tags/'+source['experiment_tag']]==source['source_git_commit']
assert not (HERE/(label+'_state.json')).exists(),'Already dispatched; do not duplicate'
if label=='t2':
    for name in ('independent_verification.json','hf_upload_verified.json','download_xet_verified.json','github_archive_verified.json'):
        assert read(HERE/'t1_results'/name)['verified']
result=remote('SOURCE='+repr(source)+'\n'+'''
import json,os,shutil,subprocess,time,sys,platform,importlib.metadata
from pathlib import Path
repo=Path(SOURCE['repository']);run=Path(SOURCE['remote_run'])
run.parent.mkdir(parents=True,exist_ok=True)
receipt=run.parent/(run.name+'_dispatch.json')
if receipt.exists():
    value=json.loads(receipt.read_text());assert value['source_git_commit']==SOURCE['source_git_commit']
else:
    assert not run.exists()
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip();assert sha==SOURCE['source_git_commit']
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert subprocess.run(['pgrep','-f','[t]rain_model.py'],capture_output=True).returncode==1
    free=shutil.disk_usage(repo).free;assert free>4_000_000_000
    fs=int(subprocess.check_output(['du','-sb','/autodl-fs/data'],text=True).split()[0]);assert fs<20_000_000_000
    with (run.parent/(run.name+'_launcher.log')).open('x') as log:
        child=subprocess.Popen(['/root/autodl-tmp/envs/betterlvit-paper/bin/python','-u','tools/run_decoder_experiment.py','--run',str(run)],
            cwd=repo,env=dict(os.environ),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    value=dict(source_git_commit=sha,repository=str(repo),remote_run=str(run),pid=child.pid,
        started_unix=time.time(),dispatch_only=True,inspections_completed=0,
        scratch_free_bytes=free,fs_bytes=fs,test_split_accessed=False,
        runtime_environment=dict(python=sys.version,platform=platform.platform(),
            packages={name:importlib.metadata.version(name) for name in ('torch','numpy','transformers','torchvision')}))
    receipt.write_text(json.dumps(value,indent=2)+'\\n')
print(json.dumps(value))
''')
result['planned_first_check_unix']=math.ceil((result['started_unix']+900)/60)*60
history=read(DOCS/'repro_archive/20260908/recipe_execution/r2_results/runtime.json')
ratio=proof['candidates'][label]['steady_seconds_per_batch']/read(HERE/'preflight/baseline.json')['steady_seconds_per_batch']
reference='historical_r2_actual_runtime_scaled_by_preflight'
if label=='t2':
    # Use the just-completed matched control to account for current throughput.
    history=read(HERE/'t1_results/runtime.json')
    assert history['phase']=='complete' and history['training_rc']==history['validation_rc']==0
    assert history['source_git_commit']==read(HERE/'sources.json')['t1']['source_git_commit']
    ratio=proof['candidates']['t2']['steady_seconds_per_batch']/proof['candidates']['t1']['steady_seconds_per_batch']
    reference='completed_t1_actual_runtime_scaled_by_t2_over_t1_preflight'
seconds=(history['training_ended_unix']-history['started_unix'])*ratio
result.update(initial_prediction_training_seconds=seconds,
    initial_predicted_training_end_unix=result['started_unix']+seconds,experiment_tag=source['experiment_tag'],
    initial_prediction_method=reference,forecast_reference_source_git_commit=history['source_git_commit'])
for key in ('started_unix','planned_first_check_unix','initial_predicted_training_end_unix'):
    result[key.replace('_unix','_sydney')]=datetime.fromtimestamp(result[key],ZoneInfo('Australia/Sydney')).isoformat()
save(label+'_launch.json',result);save(label+'_state.json',result)
print(json.dumps(result,indent=2))

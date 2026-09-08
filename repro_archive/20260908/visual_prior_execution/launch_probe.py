"""Launch the registered probe once, retain its first and final inspection budget."""
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REMOTE = r'''
import json, os, shutil, subprocess, time
from pathlib import Path
repo=Path('/root/BetterLViT-visual-prior-dev')
run=Path('/root/visual_prior_runs/probe_20260908')
cache=Path('/root/autodl-tmp/visual_probe_cache_20260908')
assert not run.exists() and not cache.exists()
assert shutil.disk_usage('/root/autodl-tmp').free > 7_000_000_000
commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
assert commit=='5d7f4eeca92462de485d972b4cd7eebb381a07b5'
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],text=True).strip()
preflights={}
for kind in ('p12_visual_prior','c9_visual_random'):
    values=[]
    for rep in (1,2):
        value=json.loads((run.parent/f'{kind}_preflight_{rep}.log').read_text().splitlines()[-1])
        assert value['status']=='ok'
        values.append(value)
    assert values[0]['output_sha256_each_step']==values[1]['output_sha256_each_step']
    assert values[0]['loss_each_step']==values[1]['loss_each_step']
    assert values[0]['first_output_sha256']=='246feaa997468b4696ac8d02772f479d38f9315814ba220fed614be7abaa39b7'
    preflights[kind]=values
timing=json.loads((run.parent/'probe_preflight.json').read_text())
env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONHASHSEED='1219')
log=run.parent/(run.name+'.log')
started=time.time()
with log.open('w') as out:
    child=subprocess.Popen(['/root/autodl-tmp/envs/betterlvit-paper/bin/python','-u','tools/probe_visual_prior.py',
        '--models','/root/visual_prior_models','--data','/root/autodl-tmp/datasets/Covid19',
        '--cache',str(cache),'--output',str(run)],cwd=repo,env=env,stdin=subprocess.DEVNULL,
        stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(2)
assert child.poll() is None, log.read_text()[-2000:]
result={'source_git_commit':commit,'remote_repository':str(repo),'remote_run':str(run),'pid':child.pid,
    'started_unix':started,'launch_inspected_unix':time.time(),'inspections_completed':1,
    'planned_final_check_unix':started+timing['planned_completion_check_delay_seconds'],
    'timing':timing,'whole_model_preflights':preflights,
    'external_weights':json.loads(Path('/root/visual_prior_models/manifest.json').read_text()),
    'test_split_accessed':False}
print(json.dumps(result))
'''

root=Path('D:/BetterLViT/outputs/visual_prior_20260908')
root.mkdir(parents=True,exist_ok=True)
assert not (root/'probe_state.json').exists(), 'Probe already launched'
key='C:/Users/dtftn/.ssh/seetacloud_betterlvit_ed25519'
process=subprocess.run(['ssh','-i',key,'-p','21465','-o','BatchMode=yes','-o','ConnectTimeout=15',
    'root@connect.westb.seetacloud.com','/root/autodl-tmp/envs/betterlvit-paper/bin/python -'],
    input=REMOTE,text=True,capture_output=True,timeout=60)
if process.returncode:
    raise RuntimeError(process.stderr)
result=json.loads(process.stdout)
result['ssh_key']=key
result['planned_final_check_sydney']=datetime.fromtimestamp(result['planned_final_check_unix'],ZoneInfo('Australia/Sydney')).isoformat()
(root/'probe_launch.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
(root/'probe_state.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:result[k] for k in ('source_git_commit','pid','started_unix','planned_final_check_sydney','inspections_completed')}))

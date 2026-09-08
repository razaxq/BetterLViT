"""Dispatch P12 once, then disconnect; schedule its first inspection separately."""
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from prepare_joint import KEY, SOURCES, remote

root=Path('D:/BetterLViT/outputs/visual_prior_20260908')
assert not (root/'p12_state.json').exists(), 'P12 already dispatched'
preflights={}
for label in ('p12','c9'):
    values=[json.loads((root/f'{label}_frozen_preflight_{rep}.json').read_text()) for rep in (1,2)]
    for value in values:
        assert value['status']=='ok' and not value['formal_training_performed']
        assert value['source_git_commit']==SOURCES[label][1]
        assert value['first_output_sha256']=='246feaa997468b4696ac8d02772f479d38f9315814ba220fed614be7abaa39b7'
        assert value['initial_base_sha256']=='6c9033efbc4c3d8d723ea24d5793833b1b84041951b9592e0274cdd8d652c1b9'
    assert values[0]['output_sha256_each_step']==values[1]['output_sha256_each_step']
    assert values[0]['loss_each_step']==values[1]['loss_each_step']
    preflights[label]=values
result=remote('''
import json, os, shutil, subprocess, time
from pathlib import Path
repo=Path('/root/BetterLViT-visual-p12')
run=Path('/root/visual_prior_runs/p12_80_20260908')
assert not run.exists()
sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert sha=='5d09913d46863073cba93159af4ed61f88fdd96e'
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
assert subprocess.run(['pgrep','-f','[t]rain_model.py'],capture_output=True).returncode!=0
assert shutil.disk_usage('/root').free > 4_000_000_000
manifest=json.loads((repo/'experiment_manifests/active_visual.json').read_text())
started=time.time()
with (run.parent/(run.name+'.log')).open('x') as log:
    child=subprocess.Popen(['/root/autodl-tmp/envs/betterlvit-paper/bin/python','-u','tools/run_visual_experiment.py',
        '--run',str(run),'--models','/root/visual_prior_models'],cwd=repo,env=dict(os.environ),
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'source_git_commit':sha,'remote_repository':str(repo),'remote_run':str(run),
    'pid':child.pid,'started_unix':started,'manifest':manifest,'inspections_completed':0,
    'dispatch_only':True,'test_split_accessed':False}))
''')
result.update(ssh_key=KEY,whole_model_preflights=preflights,
    experiment_tag='experiment-p12-visual-80e-seed1219-20260908')
result['planned_first_check_unix']=math.ceil((result['started_unix']+15*60)/60)*60
for key in ('started_unix','planned_first_check_unix'):
    result[key.replace('_unix','_sydney')]=datetime.fromtimestamp(result[key],ZoneInfo('Australia/Sydney')).isoformat()
for name in ('p12_launch.json','p12_state.json'):
    (root/name).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='whole_model_preflights'}))

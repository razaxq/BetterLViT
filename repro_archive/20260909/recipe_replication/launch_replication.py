"""Dispatch the next registered run once; receipt only, no status polling."""
import argparse
import json
import math
from datetime import datetime
from zoneinfo import ZoneInfo
from prepare_replication import HERE,ROOT,KEY,remote

p=argparse.ArgumentParser()
p.add_argument('--label',choices=('c4s2027','r2s2027','c4s3407','r2s3407'),required=True)
label=p.parse_args().label
sources=json.loads((HERE/'sources.json').read_text())
source=sources[label]
assert not (ROOT/f'{label}_state.json').exists()
proof=json.loads((ROOT/'preflight_verification.json').read_text())
assert proof['status']=='verified'
assert proof['sources']=={k:v['source_git_commit'] for k,v in sources.items()}
if source['previous_label']:
    previous=json.loads((ROOT/(source['previous_label']+'_final_snapshot.json')).read_text(encoding='utf-8'))
    assert previous['files']['runtime.json']['phase']=='complete'
    assert previous['source_git_commit']==sources[source['previous_label']]['source_git_commit']
else:
    pilot=json.loads(__import__('pathlib').Path('D:/BetterLViT/outputs/recipe_20260908/r2_summary.json').read_text())
    assert pilot['gate_passed'] and pilot['source_git_commit']=='9eca26de5b301099805530edbf5a1a8718bea662'
result=remote('SOURCE='+repr(source)+'\n'+'''
import json,os,shutil,subprocess,time
from pathlib import Path
repo=Path(SOURCE['repository'])
run=Path(SOURCE['remote_run'])
assert not run.exists()
sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert sha==SOURCE['source_git_commit']
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
assert subprocess.run(['pgrep','-f','[t]rain_model.py'],capture_output=True).returncode!=0
free=shutil.disk_usage('/root').free
assert free>4_000_000_000
manifest=json.loads((repo/'experiment_manifests/active_recipe.json').read_text())
assert manifest['profile']==SOURCE['profile'] and manifest['seed']==SOURCE['seed']
started=time.time()
with (run.parent/(run.name+'.log')).open('x') as log:
    child=subprocess.Popen(['/root/autodl-tmp/envs/betterlvit-paper/bin/python','-u','tools/run_recipe_experiment.py','--run',str(run)],
        cwd=repo,env=dict(os.environ),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(source_git_commit=sha,remote_repository=str(repo),remote_run=str(run),pid=child.pid,
    started_unix=started,manifest=manifest,inspections_completed=0,dispatch_only=True,test_split_accessed=False,system_free_bytes_before=free)))
''')
result.update(label=label,ssh_key=KEY,experiment_tag=source['experiment_tag'],preflight_verification=proof)
result['planned_first_check_unix']=math.ceil((result['started_unix']+15*60)/60)*60
for field in ('started_unix','planned_first_check_unix'):
    result[field.replace('_unix','_sydney')]=datetime.fromtimestamp(result[field],ZoneInfo('Australia/Sydney')).isoformat()
for suffix in ('launch','state'):
    (ROOT/f'{label}_{suffix}.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
chain=dict(current_label=label,order=list(sources),phase='dispatch_only',source_git_commit=source['source_git_commit'])
(ROOT/'chain_state.json').write_text(json.dumps(chain,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('preflight_verification','manifest')}))

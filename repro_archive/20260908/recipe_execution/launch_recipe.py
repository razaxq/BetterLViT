"""Dispatch one registered recipe, return its PID and disconnect without polling."""
import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from prepare_recipe import KEY, ROOT, HERE, remote
sys.path.insert(0, 'D:/BetterLViT/recipe_work')
from training_recipe import rates_equal


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--label',choices=('r1','r2'),required=True)
    label = p.parse_args().label
    sources = json.loads((HERE/'sources.json').read_text())
    source = sources[label]
    assert not (ROOT/(label+'_state.json')).exists(), 'Run already dispatched'
    if label == 'r2':
        previous = json.loads((ROOT/'r1_final_snapshot.json').read_text(encoding='utf-8'))
        runtime = previous['files'].get('runtime.json',{})
        assert runtime.get('phase') in ('complete','failed')
        assert runtime['source_git_commit'] == sources['r1']['source_git_commit']
    baseline = json.loads((ROOT/'dev_preflight_1.json').read_text())
    assert baseline['initial_base_sha256'] == '6c9033efbc4c3d8d723ea24d5793833b1b84041951b9592e0274cdd8d652c1b9'
    assert baseline['first_output_sha256'] == '246feaa997468b4696ac8d02772f479d38f9315814ba220fed614be7abaa39b7'
    checks = json.loads((ROOT/'dev_checks_2.json').read_text())
    assert checks['status']=='ok' and checks['source_git_commit']==sources['dev']['source_git_commit']
    preflights = {}
    for key in ('r1','r2'):
        values = [json.loads((ROOT/f'{key}_preflight_{rep}.json').read_text()) for rep in sources[key]['preflight_repetitions']]
        manifest = json.loads((ROOT/f'{key}_manifest.json').read_text())
        for value in values:
            assert value['status']=='ok' and not value['formal_training_performed'] and not value['test_split_accessed']
            assert value['source_git_commit']==sources[key]['source_git_commit']
            assert value['initial_base_sha256']==baseline['initial_base_sha256']
            assert value['visual_prior'] is None
            for field in ('augmentation_policy','lr_schedule','epochs'):
                assert value['training_recipe'][field]==manifest[field]
            assert rates_equal(value['training_recipe']['planned_epoch_lrs'],manifest['planned_epoch_lrs'])
        for field in ('input_image_sha256','output_sha256_each_step','loss_each_step'):
            assert values[0][field]==values[1][field]
            if key=='r2':assert values[0][field]==baseline[field]
        if key=='r1':assert values[0]['input_image_sha256']!=baseline['input_image_sha256']
        preflights[key] = values
    result = remote('SOURCE='+repr(source)+'\n'+'''
import json, os, shutil, subprocess, time
from pathlib import Path
repo=Path(SOURCE['repository'])
run=Path(SOURCE['remote_run'])
assert not run.exists()
sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert sha==SOURCE['source_git_commit']
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
assert subprocess.run(['pgrep','-f','[t]rain_model.py'],capture_output=True).returncode!=0
assert shutil.disk_usage('/root').free > 4_000_000_000
manifest=json.loads((repo/'experiment_manifests/active_recipe.json').read_text())
assert manifest['profile']==SOURCE['profile']
started=time.time()
with (run.parent/(run.name+'.log')).open('x') as log:
    child=subprocess.Popen(['/root/autodl-tmp/envs/betterlvit-paper/bin/python','-u','tools/run_recipe_experiment.py',
        '--run',str(run)],cwd=repo,env=dict(os.environ),stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(source_git_commit=sha,remote_repository=str(repo),remote_run=str(run),
    pid=child.pid,started_unix=started,manifest=manifest,inspections_completed=0,
    dispatch_only=True,test_split_accessed=False)))
''')
    result.update(ssh_key=KEY,whole_model_preflights=preflights,experiment_tag=source['experiment_tag'])
    result['planned_first_check_unix']=math.ceil((result['started_unix']+15*60)/60)*60
    for key in ('started_unix','planned_first_check_unix'):
        result[key.replace('_unix','_sydney')]=datetime.fromtimestamp(result[key],ZoneInfo('Australia/Sydney')).isoformat()
    for name in (label+'_launch.json',label+'_state.json'):
        (ROOT/name).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('whole_model_preflights','manifest')}))


if __name__=='__main__':
    main()

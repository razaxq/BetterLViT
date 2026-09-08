"""One bounded SSH snapshot per invocation; saved inspection budget prevents polling."""
import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    state = json.loads(args.state.read_text(encoding='utf-8'))
    if args.output.exists():
        raise RuntimeError('Snapshot already exists; inspect the saved file instead')
    if state['inspections_completed'] >= 2:
        raise RuntimeError('The two-inspection budget is exhausted')
    # Reserve before connecting: an uncertain connection must not create hidden retries.
    state['inspections_completed'] += 1
    args.state.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    remote = '''
import json, subprocess, time
from pathlib import Path
run = Path(RUN_VALUE).resolve()
repo = Path(REPO_VALUE).resolve()
assert run.parent == Path('/root/recipe_runs')
assert str(repo).startswith('/root/BetterLViT-recipe-')
result = {'inspected_unix':time.time(), 'run':str(run), 'repository':str(repo), 'files':{}}
timing = run / 'epoch_timing.jsonl'
if timing.exists():
    # Ignore an unfinished trailing line if the snapshot meets an epoch write.
    raw = timing.read_text()
    result['epoch_timing'] = [json.loads(line) for line in raw.splitlines()
        if line.strip() and (raw.endswith('\\n') or line != raw.splitlines()[-1])]
result['launcher_process'] = subprocess.run(
    ['ps','-p',str(PID_VALUE),'-o','pid=,stat=,etime=,args='],
    capture_output=True,text=True).stdout.strip()
for name in ('status.json','runtime.json','failure.json','validation.json',
             'cxformer_validation.json','dinov2_validation.json','cxformer_history.json','dinov2_history.json'):
    path = run / name
    if path.exists(): result['files'][name] = json.loads(path.read_text())
for name in ('training.log','validation.log','probe.log'):
    path = run / name
    if path.exists():
        with path.open('rb') as handle:
            handle.seek(max(0,path.stat().st_size-5000))
            result[name+'_tail'] = handle.read().decode('utf-8',errors='replace')
result['source_git_commit'] = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
result['tracked_changes'] = subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],text=True).strip()
runtime = result['files'].get('runtime.json',{})
launcher_log=run.parent/(run.name+'.log')
if launcher_log.exists():
    with launcher_log.open('rb') as handle:
        handle.seek(max(0,launcher_log.stat().st_size-5000))
        result['launcher_log_tail']=handle.read().decode('utf-8',errors='replace')
if 'training_ended_unix' in runtime:
    result['seconds_after_training'] = result['inspected_unix']-runtime['training_ended_unix']
probe = result['files'].get('status.json',{})
if 'encoders' in probe:
    result['seconds_after_each_probe'] = {k:result['inspected_unix']-v['completed_unix']
        for k,v in probe['encoders'].items() if 'completed_unix' in v}
print(json.dumps(result))
'''.replace('RUN_VALUE',repr(state['remote_run'])).replace('REPO_VALUE',repr(state['remote_repository'])).replace('PID_VALUE',repr(state['pid']))
    process = subprocess.run(['ssh','-i',state['ssh_key'],'-p','21465','-o','BatchMode=yes',
        '-o','ConnectTimeout=15','root@connect.westb.seetacloud.com',
        '/root/autodl-tmp/envs/betterlvit-paper/bin/python -'],
        input=remote,text=True,capture_output=True,timeout=60)
    if process.returncode:
        raise RuntimeError(process.stderr)
    result = json.loads(process.stdout)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    state['last_snapshot'] = str(args.output.resolve())
    state['last_inspected_unix'] = result['inspected_unix']
    args.state.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    # Console code pages can reject progress-bar glyphs after a successful save.
    # Escape console text; the complete on-disk UTF-8 snapshot remains unchanged.
    print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=True))


if __name__=='__main__':
    main()

"""Launch once, then make at most two short predicted Test completion inspections."""
import argparse
import base64
import hashlib
import json
import math
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from protocol import load_plan, sha256, validate_result, write_json

HERE = Path(__file__).resolve().parent
DOCS = HERE.parents[2]
OUT = Path('D:/BetterLViT/outputs/recipe_test_20260910')
REMOTE = '/root/recipe_test_20260910'
PYTHON = '/root/autodl-tmp/envs/betterlvit-paper/bin/python'
KEY = 'C:/Users/dtftn/.ssh/seetacloud_betterlvit_ed25519'
SOURCES = ('evaluate_test.py', 'protocol.py', 'compare_test.py', 'run_test_chain.py',
           'test_plan.json', 'three_seed_summary.json')


def remote(code):
    result = subprocess.run(['ssh', '-i', KEY, '-p', '21465', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                             'root@connect.westb.seetacloud.com', PYTHON + ' -'], input=code, text=True,
                            encoding='utf-8', capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr + result.stdout[-4000:])
    return json.loads(result.stdout)


def launch():
    plan = load_plan(HERE)
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=DOCS, text=True).strip(), 'Commit first'
    analysis_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=DOCS, text=True).strip()
    payload = {}
    for name in SOURCES:
        path = (HERE / name).relative_to(DOCS).as_posix()
        blob = subprocess.check_output(['git', 'show', analysis_sha + ':' + path], cwd=DOCS)
        assert blob == (HERE / name).read_bytes(), 'Working bytes differ from committed source'
        payload[name] = base64.b64encode(blob).decode('ascii')
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'launch_attempt.json').exists(), 'No blind launch retry; reconcile receipt first'
    write_json(OUT / 'launch_attempt.json', dict(started_unix=time.time(), evaluation_source_git_commit=analysis_sha))
    result = remote('PAYLOAD=' + repr(payload) + '\nSHA=' + repr(analysis_sha) + '\nROOT=' + repr(REMOTE) + '\n' + '''
import base64, hashlib, json, os, shutil, subprocess, time
from pathlib import Path
root=Path(ROOT)
assert not root.exists(), 'Test directory already exists; do not overwrite or rerun'
free=shutil.disk_usage('/root').free
assert free >= 1_000_000_000, 'Insufficient system disk space for Test artifacts'
shared=int(subprocess.check_output(['du','-sb','/autodl-fs/data'],text=True).split()[0])
assert shared < 20_000_000_000
root.mkdir()
for name, encoded in PAYLOAD.items():
    assert Path(name).name==name
    (root/name).write_bytes(base64.b64decode(encoded))
plan=json.loads((root/'test_plan.json').read_text())
for arm in plan['arms']:
    repo=arm['repository']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==arm['source_git_commit']
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
    assert Path(arm['checkpoint']).is_file()
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True).strip()
started=time.time()
with (root/'chain.log').open('x') as log:
    process=subprocess.Popen(['/root/autodl-tmp/envs/betterlvit-paper/bin/python','-u',str(root/'run_test_chain.py'),str(root),SHA],
        cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
receipt=dict(pid=process.pid,submitted_unix=started,evaluation_source_git_commit=SHA,remote_directory=str(root),
    system_free_bytes=free,shared_fs_bytes=shared,gpu_at_submission=gpu,
    deployed_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in PAYLOAD})
(root/'launch_receipt.json').write_text(json.dumps(receipt,indent=2)+chr(10))
print(json.dumps(receipt))
''')
    for name in SOURCES:
        assert result['deployed_sha256'][name] == sha256(HERE / name)
    write_json(OUT / 'launch.json', result)
    expected = result['submitted_unix'] + plan['forecast']['estimated_chain_seconds']
    check = math.ceil((expected + 720) / 60) * 60
    state = dict(phase='submitted', inspections_completed=0, maximum_inspections=2,
                 evaluation_source_git_commit=analysis_sha, expected_completion_unix=expected,
                 expected_completion_sydney=datetime.fromtimestamp(expected, ZoneInfo('Australia/Sydney')).isoformat(),
                 planned_check_unix=check, planned_check_sydney=datetime.fromtimestamp(check, ZoneInfo('Australia/Sydney')).isoformat(),
                 no_persistent_ssh=True, training_inspections_unchanged=True)
    write_json(OUT / 'state.json', state)
    print(json.dumps(dict(launch=result, state=state)))


def collect():
    state = json.loads((OUT / 'state.json').read_text(encoding='utf-8'))
    assert state['phase'] not in ('complete', 'failed') and state['inspections_completed'] < 2
    assert time.time() >= state['planned_check_unix'], 'Wait for predicted final inspection'
    state['inspections_completed'] += 1
    state['last_inspection_started_unix'] = time.time()
    write_json(OUT / 'state.json', state)
    snapshot = remote('ROOT=' + repr(REMOTE) + '\n' + '''
import base64, hashlib, json, time
from pathlib import Path
root=Path(ROOT)
runtime=json.loads((root/'runtime.json').read_text()) if (root/'runtime.json').exists() else dict(phase='missing_runtime')
snapshot=dict(observed_unix=time.time(),runtime=runtime,files={})
if runtime['phase']=='complete':
    for name, meta in runtime['artifacts'].items():
        data=(root/name).read_bytes()
        assert len(data)==meta['bytes'] and hashlib.sha256(data).hexdigest()==meta['sha256']
        snapshot['files'][name]=base64.b64encode(data).decode('ascii')
elif runtime['phase'] in ('failed','missing_runtime'):
    snapshot['log_tails']={p.name:p.read_text(errors='replace')[-8000:] for p in root.glob('*.log')}
print(json.dumps(snapshot))
''')
    payload = snapshot.pop('files')
    write_json(OUT / f'inspection_{state["inspections_completed"]}.json', snapshot)
    state['phase'] = snapshot['runtime']['phase']
    if state['phase'] == 'complete':
        plan = load_plan(HERE)
        for name, encoded in payload.items():
            assert Path(name).name == name
            data = base64.b64decode(encoded)
            assert hashlib.sha256(data).hexdigest() == snapshot['runtime']['artifacts'][name]['sha256']
            assert not (OUT / name).exists()
            (OUT / name).write_bytes(data)
        for arm in plan['arms']:
            validate_result(json.loads((OUT / (arm['label'] + '_test.json')).read_text()), arm, plan,
                            state['evaluation_source_git_commit'], sha256(HERE / 'evaluate_test.py'))
        state['seconds_after_test_completion'] = snapshot['observed_unix'] - snapshot['runtime']['completed_unix']
        state['completion_check_within_30min'] = 0 <= state['seconds_after_test_completion'] <= 1800
    write_json(OUT / 'state.json', state)
    print(json.dumps(dict(state=state, runtime=snapshot['runtime'], downloaded_files=len(payload))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('launch', 'collect'))
    args = parser.parse_args()
    launch() if args.action == 'launch' else collect()

"""Short SSH deployment and predicted collection; no persistent connection."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
DOCS = HERE.parents[2]
sys.path.insert(0, str(HERE.parents[1] / '20260910/visual_aux_execution'))
from remote_ops import remote

REMOTE = '/root/autodl-tmp/visual_aux_test_20260911'


def launch():
    assert not (HERE / 'launch_attempt.json').exists(), 'Reconcile existing launch before retrying'
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=DOCS, text=True).strip(), 'Commit first'
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=DOCS, text=True).strip()
    files = {}
    for name in ('evaluate_test.py', 'run_chain.py', 'authorization.json'):
        blob = subprocess.check_output(['git', 'show', sha + ':' + (HERE / name).relative_to(DOCS).as_posix()], cwd=DOCS)
        files[name] = base64.b64encode(blob).decode()
    (HERE / 'launch_attempt.json').write_text(json.dumps(dict(time_unix=time.time(), evaluation_source_git_commit=sha)))
    value = remote('FILES=' + repr(files) + '\nSHA=' + repr(sha) + '\nROOT=' + repr(REMOTE) + '\n' + '''
import base64,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
root=Path(ROOT)
assert not root.exists(), 'Existing evaluation directory; do not overwrite'
assert shutil.disk_usage('/root/autodl-tmp').free>500_000_000
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(), 'GPU occupied'
root.mkdir()
for name,value in FILES.items():(root/name).write_bytes(base64.b64decode(value))
with (root/'chain.log').open('x') as log:
    p=subprocess.Popen(['/root/autodl-tmp/envs/betterlvit-paper/bin/python','-u',str(root/'run_chain.py'),str(root),SHA],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
value=dict(pid=p.pid,submitted_unix=time.time(),evaluation_source_git_commit=SHA,remote_directory=ROOT,training_started=False,deployed_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in FILES})
(root/'launch.json').write_text(json.dumps(value,indent=2)+'\\n')
print(json.dumps(value))
''')
    plan = json.loads((HERE / 'authorization.json').read_text(encoding='utf-8'))
    value['planned_collection_unix'] = value['submitted_unix'] + plan['planned_collection_seconds_after_launch']
    (HERE / 'launch.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value))


def collect():
    launch = json.loads((HERE / 'launch.json').read_text())
    assert time.time() >= launch['planned_collection_unix'], 'Use predicted collection time'
    assert not (HERE / 'collection.json').exists(), 'Results already collected'
    value = remote('ROOT=' + repr(REMOTE) + '\n' + '''
import base64,hashlib,json,time
from pathlib import Path
root=Path(ROOT);state=json.loads((root/'runtime.json').read_text())
value=dict(observed_unix=time.time(),runtime=state,files={})
if state['phase']=='complete':
    for name,meta in state['artifacts'].items():
        blob=(root/name).read_bytes()
        assert hashlib.sha256(blob).hexdigest()==meta['sha256'] and len(blob)==meta['bytes']
        value['files'][name]=base64.b64encode(blob).decode()
elif state['phase']=='failed':
    value['log_tails']={p.name:p.read_text(errors='replace')[-6000:] for p in root.glob('*.log')}
print(json.dumps(value))
''')
    payload = value.pop('files')
    (HERE / 'collection_snapshot.json').write_text(json.dumps(value, indent=2) + '\n')
    if value['runtime']['phase'] != 'complete':
        print(json.dumps(value));return
    destination = HERE / 'results';destination.mkdir(exist_ok=True)
    for name, encoded in payload.items():
        assert Path(name).name == name
        blob = base64.b64decode(encoded)
        assert hashlib.sha256(blob).hexdigest() == value['runtime']['artifacts'][name]['sha256']
        (destination / name).write_bytes(blob)
    value['seconds_after_completion'] = value['observed_unix'] - value['runtime']['completed_unix']
    (HERE / 'collection.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(phase='complete',downloaded_files=len(payload),seconds_after_completion=value['seconds_after_completion'])))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('launch','collect'))
    {'launch':launch,'collect':collect}[parser.parse_args().action]()

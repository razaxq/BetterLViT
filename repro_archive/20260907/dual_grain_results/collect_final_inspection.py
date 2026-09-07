"""Collect one final, read-only SSH snapshot; never poll or restart a run."""
import json
from pathlib import Path
import subprocess

REMOTE = r'''
import json
from pathlib import Path
import subprocess
import time

run = Path('/root/dual_grain_runs/p11_20260907')
inspected = time.time()
result = {'inspected_unix': inspected, 'run': str(run), 'files': {}, 'status': {}}
for name in ('chain', 'training', 'validation', 'test', 'compare_validation', 'compare_test'):
    path = run / (name + '.status')
    result['status'][name] = None if not path.exists() else {
        'value': path.read_text().strip(), 'modified_unix': path.stat().st_mtime}
for name in ('runtime.json', 'p11_validation.json', 'p11_test.json',
             'c4_vs_p11_validation.json', 'c4_vs_p11_test.json'):
    path = run / name
    result['files'][name] = json.loads(path.read_text()) if path.exists() else None
runtime = result['files']['runtime.json']
repo = runtime['repository']
assert repo == '/root/autodl-tmp/BetterLViT-dual-grain-p11'
result['repository_commit'] = subprocess.check_output(['git', '-C', repo, 'rev-parse', 'HEAD'], text=True).strip()
result['tracked_source_changes'] = subprocess.check_output(
    ['git', '-C', repo, 'status', '--porcelain', '--untracked-files=no'], text=True)
for name in ('training', 'validation', 'test', 'compare_validation', 'compare_test'):
    path = run / (name + '.log')
    if path.exists():
        with path.open('rb') as handle:
            handle.seek(max(0, path.stat().st_size - 5000))
            result[name + '_log_tail'] = handle.read().decode('utf-8', errors='replace')
print(json.dumps(result))
'''

root = Path(__file__).resolve().parent
destination = root / 'final_inspection.json'
assert not destination.exists(), 'Do not perform a duplicate final inspection'
process = subprocess.run(['ssh', '-i', 'C:/Users/dtftn/.ssh/seetacloud_betterlvit_ed25519_4090d',
    '-p', '21465', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
    'root@connect.westb.seetacloud.com', '/root/autodl-tmp/envs/betterlvit-paper/bin/python -'],
    input=REMOTE, text=True, capture_output=True, timeout=90)
if process.returncode:
    print(process.stderr)
    raise SystemExit(process.returncode)
data = json.loads(process.stdout)
destination.write_text(json.dumps(data, indent=2), encoding='utf-8')
summary = {key: data[key] for key in ('inspected_unix', 'status', 'repository_commit', 'tracked_source_changes')}
summary['test'] = {k: v for k, v in (data['files']['p11_test.json'] or {}).items() if k != 'records'}
summary['comparison'] = data['files']['c4_vs_p11_test.json']
summary['comparison_log'] = data.get('compare_test_log_tail')
print(json.dumps(summary, indent=2))

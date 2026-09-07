"""One brief launch inspection; do not use this as a recurring monitor."""
import json
from pathlib import Path
import subprocess

REMOTE = r'''
import json, subprocess, time
from pathlib import Path
run = Path('/root/dual_grain_runs/p11_150_20260907')
preflight = Path('/root/dual_grain_150_preflight_20260907')
result = {'inspected_unix': time.time(), 'files': {}}
for name, path in [('runtime.json', run/'runtime.json'), ('launch.json', preflight/'launch.json'),
                   ('first.json', preflight/'first.json'), ('repeat.json', preflight/'repeat.json')]:
    result['files'][name] = json.loads(path.read_text()) if path.exists() else None
result['statuses'] = {p.name: p.read_text().strip() for p in run.glob('*.status')}
for name, path in [('training', run/'training.log'), ('launcher', preflight/'launcher.log')]:
    result[name+'_log_tail'] = path.read_text(errors='replace')[-7000:] if path.exists() else None
result['gpu'] = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], text=True)
result['source_git_commit'] = subprocess.check_output(['git', '-C', '/root/autodl-tmp/BetterLViT-dual-grain-p11-150', 'rev-parse', 'HEAD'], text=True).strip()
print(json.dumps(result))
'''
destination = Path('D:/BetterLViT/outputs/dual_grain_150_monitor_20260907/launch_inspection.json')
assert not destination.exists(), 'Do not repeat the launch inspection'
process = subprocess.run(['ssh', '-i', 'C:/Users/dtftn/.ssh/seetacloud_betterlvit_ed25519_4090d',
    '-p', '21465', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
    'root@connect.westb.seetacloud.com', '/root/autodl-tmp/envs/betterlvit-paper/bin/python -'],
    input=REMOTE, text=True, capture_output=True, timeout=45, check=True)
data = json.loads(process.stdout)
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(data, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in data.items() if k != 'files'}, indent=2))

"""Run on the server to launch P11 continuation, detached from SSH."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

repo = Path('/root/autodl-tmp/BetterLViT-dual-grain-p11-150')
preflight = Path('/root/dual_grain_150_preflight_20260907')
output = Path('/root/dual_grain_runs/p11_150_20260907')
expected = 'c724a62001f6c3b809cb12eee78f3bee79bdb60a'
assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == expected
assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'])
a, b = [json.loads((preflight / (name + '.json')).read_text()) for name in ('first', 'repeat')]
assert a == b and a['status'] == 'ok' and a['model_optimizer_scheduler_rng_restored']
# Only a comparison regression test was added after the resume preflight.
changed = subprocess.check_output(['git', '-C', str(repo), 'diff', '--name-only',
    'd7810def636acf61b68d26708a893b36d10be2a6', expected], text=True).splitlines()
assert changed == ['tools/test_p11_comparison.py']
assert not output.exists()
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
with (preflight / 'launcher.log').open('w') as log:
    process = subprocess.Popen(['/root/autodl-tmp/envs/betterlvit-paper/bin/python',
        str(repo / 'tools/run_p11_continuation.py'), '--parent-run',
        '/root/dual_grain_runs/p11_20260907', '--output', str(output)], cwd=repo,
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
proof = {'pid': process.pid, 'source_git_commit': expected, 'output': str(output),
    'tag': 'paper-p11-150e-resume-b16-seed1219-20260907',
    'launched_utc': datetime.now(timezone.utc).isoformat(), 'resume_preflight_identical': True}
(preflight / 'launch.json').write_text(json.dumps(proof, indent=2))
print(json.dumps(proof, indent=2))

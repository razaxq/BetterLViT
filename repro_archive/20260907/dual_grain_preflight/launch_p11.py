"""Launch the verified frozen P11 runner independently of SSH lifetime."""
import json
from pathlib import Path
import subprocess

repo = Path('/root/autodl-tmp/BetterLViT-dual-grain-p11')
preflight = Path('/root/dual_grain_preflight_20260907')
output = Path('/root/dual_grain_runs/p11_20260907')
python = '/root/autodl-tmp/envs/betterlvit-paper/bin/python'
expected = '2fc6ab5c8e4662d741fd8b994e55b780391948ac'
assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == expected
first, repeat = [json.loads((preflight / name).read_text()) for name in ('first.json', 'repeat.json')]
assert first['status'] == repeat['status'] == 'ok'
assert first['real_batch']['output_sha256'] == repeat['real_batch']['output_sha256']
assert first['real_batch']['losses'] == repeat['real_batch']['losses']
assert not output.exists()
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
output.parent.mkdir(parents=True, exist_ok=True)
with (preflight / 'launcher.log').open('w') as log:
    process = subprocess.Popen([python, str(repo / 'tools/run_dual_grain_experiment.py'),
        '--output', str(output),
        '--control-validation', '/root/race_pe_runs/c4_p9_20260906/c4_validation.json',
        '--control-test', '/root/race_pe_test_20260907/c4_test.json'], cwd=repo,
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
proof = {'launcher_pid': process.pid, 'source_git_commit': expected, 'output': str(output),
         'tag': 'paper-p11-dual-grain-80e-b16-seed1219-20260907', 'preflight_repeat_identical': True}
(preflight / 'launch.json').write_text(json.dumps(proof, indent=2), encoding='utf-8')
print(json.dumps(proof, indent=2))

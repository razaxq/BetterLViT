"""Preserve an old, previously uncommitted server draft without its dataset link."""
import ast
import re
import subprocess
from pathlib import Path

root = Path('/root/autodl-tmp/BetterLViT-paper-tcsrv2-dev')
paths = ['Config.py', 'nets/LViT.py', 'paper_experiments.py',
         'scripts/run_paper_experiment.ps1', 'scripts/start_paper_experiment.ps1',
         'scripts/start_paper_experiment_server.sh', 'tools/evaluate_experiment.py',
         'tools/smoke_paper_profile.py', 'train_model.py', 'docs/TCSRV2_DESIGN.md',
         'docs/TCSR_DESIGN.md', 'experiment_manifests/a8_tcsrv2_freq_focal.json',
         'nets/tcsr.py', 'tools/check_tcsr.py']
for name in paths:
    data = (root / name).read_text(encoding='utf-8-sig')
    assert not re.search(r'hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|BEGIN .*PRIVATE KEY', data), name
    if name.endswith('.py'):
        ast.parse(data, filename=name)
def git(*args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True)
assert git('rev-parse', 'HEAD').strip() == '494ec30ce1cd94fac566b480877ea3391f5d64ed'
branch = 'archive/server-tcsrv2-draft-20260907'
if git('branch', '--show-current').strip() != branch:
    git('switch', '-c', branch)
note = 'docs/SERVER_DRAFT_ARCHIVE_20260907.md'
(root / note).write_text('''# Historical server draft snapshot

Archived on 2026-09-07 from the previously dirty server worktree
`/root/autodl-tmp/BetterLViT-paper-tcsrv2-dev`, based on
`494ec30ce1cd94fac566b480877ea3391f5d64ed`.

This preserves the exact old TCSR V2 source draft for audit and reproducibility.
It is not a new experiment, a tested replacement for later frozen branches, or
evidence of a performance gain. Only Python parsing and credential-pattern
checks were performed when archiving. The untracked dataset symlink is excluded.
Historical CRLF/trailing whitespace is intentionally retained in this snapshot.
''', encoding='utf-8')
git('add', '--', *paths, note)
print(git('commit', '-m', 'archive: preserve historical server TCSR V2 draft'))
print(git('rev-parse', 'HEAD'))
print(git('bundle', 'create', '/root/maintenance_20260907/server_draft.bundle', branch))

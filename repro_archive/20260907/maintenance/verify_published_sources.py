"""Prove all audited server source commits are reachable from published refs."""
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
repo = 'D:/BetterLViT/tcsr_work'
def git(*args):
    return subprocess.check_output(['git', '-C', repo, *args], text=True).strip()
remote = dict(line.split()[::-1] for line in git('ls-remote', 'https://github.com/razaxq/BetterLViT.git').splitlines())
audited = json.loads((root / 'remote_source_audit_after.json').read_text())
proof = []
for item in audited:
    refs = git('for-each-ref', '--contains=' + item['commit'], '--format=%(objectname) %(refname)',
               'refs/heads', 'refs/tags').splitlines()
    matches = []
    for line in refs:
        oid, ref = line.split()
        if remote.get(ref) == oid:
            matches.append(ref)
    assert matches, item['path']
    assert item['status'] in ('', '?? datasets\n'), item['path']
    proof.append({'path': item['path'], 'commit': item['commit'], 'published_refs': matches})
(root / 'published_sources_verified.json').write_text(json.dumps(proof, indent=2), encoding='utf-8')
print('SERVER_SOURCES_PUBLISHED', len(proof))

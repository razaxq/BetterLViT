"""Verify exact committed raw bytes and source tag before delivery."""
import hashlib
import json
from pathlib import Path
import subprocess
HERE=Path(__file__).resolve().parent
DOCS=next(p for p in HERE.parents if (p/'.git').exists())
runtime=json.loads((HERE/'results/runtime.json').read_text())
assert subprocess.check_output(['git','rev-parse','diagnostic-text-iou-e2-20260912'],cwd=DOCS,text=True).strip()==runtime['source_git_commit']
for name,meta in runtime['artifacts'].items():
    path=(HERE/'results'/name).relative_to(DOCS).as_posix()
    committed=subprocess.check_output(['git','show','HEAD:'+path],cwd=DOCS)
    assert hashlib.sha256(committed).hexdigest()==meta['sha256'],name
    assert committed==(HERE/'results'/name).read_bytes(),name
print(json.dumps(dict(verified=True,raw_git_artifacts=len(runtime['artifacts']),
    source_git_commit=runtime['source_git_commit'],report_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=DOCS,text=True).strip())))

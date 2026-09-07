"""Collect small reproducibility artifacts, never credentials or checkpoints."""
import ast
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path('D:/BetterLViT')
DEST = ROOT / 'experiment_docs_work/repro_archive/20260907'
SECRET = re.compile(rb'hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY-----')
EXTENSIONS = {'.py', '.sh', '.ps1', '.md', '.json', '.txt', '.status'}
folders = ['race_pe_results_20260907', 'race_pe_test_20260907',
           'race_pe_diagnosis_20260907', 'race_pe_v2_launch_20260907',
           'race_pe_v2_probe_20260907', 'race_pe_v2_results_20260907',
           'race_pe_v2_test_20260907', 'research_20260907']
copied = []
for folder in folders:
    for src in (ROOT / 'outputs' / folder).rglob('*'):
        if not src.is_file() or src.suffix not in EXTENSIONS or '__pycache__' in src.parts:
            continue
        data = src.read_bytes()
        if len(data) > 10_000_000:
            raise RuntimeError(f'Unexpected large text artifact: {src}')
        if SECRET.search(data):
            raise RuntimeError(f'Credential pattern in {src}')
        if src.suffix == '.py':
            ast.parse(data.decode('utf-8-sig'), filename=str(src))
        target = DEST / folder / src.relative_to(ROOT / 'outputs' / folder)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        copied.append(str(target.relative_to(DEST)))
for src in (ROOT / 'tools').iterdir():
    if src.suffix not in {'.py', '.sh', '.ps1'}:
        continue
    data = src.read_bytes()
    if SECRET.search(data):
        raise RuntimeError(f'Credential pattern in {src}')
    if src.suffix == '.py':
        ast.parse(data.decode('utf-8-sig'), filename=str(src))
    target = DEST / 'server_operations' / src.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    copied.append(str(target.relative_to(DEST)))
(DEST / 'inventory.json').write_text(json.dumps(copied, indent=2), encoding='utf-8')
print(json.dumps({'archived_files': len(copied), 'destination': str(DEST)}))

# Validate pending historical snapshots without executing training code.
for folder in ['BetterLViT-Migrated', 'BetterLViT-4090D-src']:
    repo = ROOT / folder
    names = set(subprocess.check_output(['git','-C',str(repo),'diff','--name-only'], text=True).splitlines())
    names.update(subprocess.check_output(['git','-C',str(repo),'ls-files','--others','--exclude-standard'], text=True).splitlines())
    for name in names:
        src = repo / name
        if not src.is_file():
            continue
        data = src.read_bytes()
        if SECRET.search(data):
            raise RuntimeError(f'Credential pattern in {src}')
        if src.suffix == '.py':
            ast.parse(data.decode('utf-8-sig'), filename=str(src))
    print(folder, 'pending files checked', len(names))

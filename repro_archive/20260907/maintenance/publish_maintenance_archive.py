"""Copy only credential-free maintenance source and audit JSON into Git docs."""
import ast
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
dest = Path('D:/BetterLViT/experiment_docs_work/repro_archive/20260907/maintenance')
dest.mkdir(parents=True, exist_ok=True)
names = []
for path in sorted(root.iterdir()):
    if not path.is_file() or path.suffix not in {'.py', '.json', '.md'}:
        continue
    data = path.read_bytes()
    assert len(data) < 5_000_000, path.name
    assert not re.search(rb'hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY-----', data), path.name
    if path.suffix == '.py':
        ast.parse(data.decode('utf-8-sig'), filename=path.name)
    elif path.suffix == '.json':
        json.loads(data)
    (dest / path.name).write_bytes(data)
    names.append(path.name)
print(json.dumps({'files': names, 'count': len(names)}, ensure_ascii=False))

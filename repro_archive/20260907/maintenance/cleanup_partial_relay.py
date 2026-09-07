"""Remove only this task's incomplete relay files after full HF verification."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
proof = json.loads((root / 'new_runs_upload_verified.json').read_text())
assert len(proof) == 4 and all(x['verified'] for x in proof)
removed = []
for arm in proof:
    for item in arm['files']:
        candidates = [root / 'local_model_backup' / item['remote'],
                      root / 'prefetched' / (item['remote'] + '.partial')]
        for path in candidates:
            assert path.resolve().is_relative_to(root)
            if path.is_file() and (path.name.endswith('.partial') or path.stat().st_size != item['bytes']):
                removed.append({'path': str(path), 'bytes': path.stat().st_size, 'hf_verified': item['remote']})
                path.unlink()
(root / 'partial_relay_cleanup.json').write_text(json.dumps(removed, indent=2), encoding='utf-8')
print('INCOMPLETE_RELAY_FILES_REMOVED', len(removed), sum(x['bytes'] for x in removed))

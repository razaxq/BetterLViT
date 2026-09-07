"""Resume verified-source uploads with bounded Xet concurrency."""
import json
import os
from pathlib import Path
import sys

os.environ['HF_TOKEN'] = sys.stdin.readline().strip()
os.environ['HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY'] = 'false'
os.environ['HF_XET_CLIENT_AC_INITIAL_UPLOAD_CONCURRENCY'] = '1'
os.environ['HF_XET_FIXED_UPLOAD_CONCURRENCY'] = '1'
os.environ['HF_XET_DEDUPLICATION_GLOBAL_DEDUP_QUERY_ENABLED'] = 'false'
os.environ['HF_XET_CACHE'] = '/root/maintenance_20260907/xet_serial_cache'
os.environ['HF_XET_LOG_LEVEL'] = 'info'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
from huggingface_hub import HfApi

root = Path('/root/maintenance_20260907')
manifest = json.loads((root / 'transfer_manifest.json').read_text())
api = HfApi()
bucket = 'razaxq/BetterLViT'
verified = []
for arm in manifest:
    prefix = arm['source_git_commit'][:8]
    for item in arm['files']:
        remote = {x.path: x for x in api.list_bucket_tree(bucket, prefix=prefix, recursive=True)
                  if getattr(x, 'xet_hash', None)}
        existing = remote.get(item['remote'])
        if existing:
            assert existing.size == item['bytes'] and existing.xet_hash == item['xet_hash']
        else:
            assert Path(item['source']).stat().st_size == item['bytes']
            print('UPLOAD_BEGIN', item['remote'], flush=True)
            api.batch_bucket_files(bucket, add=[(Path(item['source']).read_bytes(), item['remote'])])
        remote = {x.path: x for x in api.list_bucket_tree(bucket, prefix=prefix, recursive=True)
                  if getattr(x, 'xet_hash', None)}
        got = remote[item['remote']]
        assert got.size == item['bytes'] and got.xet_hash == item['xet_hash']
        print('HF_VERIFIED', item['remote'], flush=True)
    assert set(remote) == {item['remote'] for item in arm['files']}
    arm['verified'] = True
    arm['classification'] = 'completed_exploratory_80e'
    verified.append(arm)
    (root / 'new_runs_upload_verified.json').write_text(json.dumps(verified, indent=2), encoding='utf-8')
print('SERIAL_COMPLETE', flush=True)

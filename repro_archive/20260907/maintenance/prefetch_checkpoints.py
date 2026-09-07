"""Fetch remaining checkpoints concurrently into atomic, verified local files."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import hf_xet

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'transfer_manifest.json').read_text())
items = [x for a in manifest[1:] for x in a['files'] if '/models/' in x['remote']]
def fetch(item):
    target = root / 'prefetched' / item['remote']
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        partial = target.with_suffix(target.suffix + '.partial')
        print('PREFETCH_BEGIN', item['remote'], flush=True)
        subprocess.run(['scp', '-i', 'C:/Users/dtftn/.ssh/seetacloud_betterlvit_ed25519_4090d',
                        '-P', '21465', '-o', 'BatchMode=yes', '-o', 'ServerAliveInterval=15',
                        'root@connect.westb.seetacloud.com:' + item['source'], str(partial)], check=True)
        assert partial.stat().st_size == item['bytes']
        partial.replace(target)
    assert hf_xet.hash_files([str(target)])[0].hash == item['xet_hash']
    print('PREFETCH_VERIFIED', item['remote'], flush=True)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(fetch, items))

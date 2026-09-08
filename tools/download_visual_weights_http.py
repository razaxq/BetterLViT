"""Standard-library fallback for pinned public weights; verifies official LFS SHA256."""
import argparse
import hashlib
import json
import shutil
import time
import urllib.request
from pathlib import Path

MODELS = {'cxformer': ('m42-health/CXformer-small', '21777c302a43197b704c6c92591d9897efaaac3b'),
          'dinov2': ('facebook/dinov2-with-registers-small', '0d9846e56b43a21fa46d7f3f5070f0506a5795a9')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    manifest = {}
    for kind, (model_id, revision) in MODELS.items():
        api = f'https://huggingface.co/api/models/{model_id}/revision/{revision}?blobs=true'
        with urllib.request.urlopen(api, timeout=30) as response:
            metadata = json.load(response)
        assert metadata['sha'] == revision
        entries = {v['rfilename']: v for v in metadata['siblings']}
        folder = args.output / kind
        folder.mkdir(parents=True, exist_ok=True)
        manifest[kind] = {'model_id': model_id, 'revision': revision, 'files': {}}
        for name in ('config.json', 'preprocessor_config.json', 'model.safetensors'):
            path = folder / name
            if not path.exists():
                url = f'https://huggingface.co/{model_id}/resolve/{revision}/{name}?download=true&nonce={time.time_ns()}'
                with urllib.request.urlopen(url, timeout=45) as response, path.with_suffix('.part').open('wb') as output:
                    shutil.copyfileobj(response, output, 2**20)
                path.with_suffix('.part').replace(path)
            with path.open('rb') as handle:
                digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            if 'lfs' in entries[name]:
                assert digest == entries[name]['lfs']['sha256'], 'Official LFS checksum mismatch'
                assert path.stat().st_size == entries[name]['lfs']['size']
            manifest[kind]['files'][name] = digest
            print(kind, name, path.stat().st_size, flush=True)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()

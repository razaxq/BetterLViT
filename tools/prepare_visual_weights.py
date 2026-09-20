"""Download only pinned public safetensors/configs; no remote Python execution."""
import argparse
import json
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nets.frozen_visual import ENCODERS, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    manifest = {}
    for kind, (model_id, revision) in ENCODERS.items():
        folder = args.output / kind
        manifest[kind] = {'model_id': model_id, 'revision': revision, 'files': {}}
        for name in ('config.json', 'preprocessor_config.json', 'model.safetensors'):
            path = hf_hub_download(model_id, name, revision=revision, local_dir=folder, token=False)
            manifest[kind]['files'][name] = sha256(path)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()

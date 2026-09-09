"""Prepare and upload the remaining completed P11/P12 training archives."""
import argparse
import base64
import json
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, get_token
from upload_pending import BUCKET, DOCS, HERE, OUT, sanitized, ssh, write_json

REMOTE = '/root/hf_recovery_20260910/v2'


def prepare():
    files = ('prepare_additional_server.py', 'upload_verified_manifest.py', 'additional_plan.json')
    payload = {name: base64.b64encode((HERE / name).read_bytes()).decode('ascii') for name in files}
    code = 'PAYLOAD=' + repr(payload) + '\n' + '''
import base64,json,subprocess
from pathlib import Path
root=Path('/root/hf_recovery_20260910/v2')
root.mkdir(exist_ok=True,parents=True)
for name, encoded in PAYLOAD.items():
    data=base64.b64decode(encoded)
    path=root/name
    if path.exists():assert path.read_bytes()==data
    else:path.write_bytes(data)
result=subprocess.run(['ionice','-c','3','nice','-n','10','/root/autodl-tmp/envs/betterlvit-paper/bin/python',str(root/'prepare_additional_server.py')],capture_output=True,text=True)
assert result.returncode==0,result.stderr+result.stdout[-3000:]
print(result.stdout)
'''
    result = ssh('/root/autodl-tmp/envs/betterlvit-paper/bin/python -', input=code, timeout=240)
    if result.returncode:
        raise RuntimeError(result.stderr + result.stdout[-4000:])
    manifests = json.loads(result.stdout)
    write_json(OUT / 'additional_manifests.json', manifests)
    print(json.dumps(dict(prepared=len(manifests), files=sum(len(m['files']) for m in manifests),
                          source_bytes=sum(f['bytes'] for m in manifests for f in m['files']))))


def upload():
    assert json.loads((OUT / 'state.json').read_text())['phase'] == 'complete', 'Complete recipe upload batch first'
    manifests = json.loads((OUT / 'additional_manifests.json').read_text())
    token = get_token()
    assert token
    api = HfApi(token=token)
    proofs = []
    for manifest in manifests:
        label, prefix = manifest['label'], manifest['bucket_prefix']
        print(json.dumps(dict(event='additional_upload_started', label=label, prefix=prefix)), flush=True)
        result = ssh('ionice -c 3 nice -n 10 env PYTHONPATH=/root/autodl-tmp/hf-bucket-client '
                     '/root/autodl-tmp/envs/betterlvit-paper/bin/python ' + REMOTE + '/upload_verified_manifest.py '
                     '--manifest ' + REMOTE + '/' + prefix + '_manifest.json', input=token + '\n', timeout=1200)
        (OUT / (label + '_upload.log')).write_text(sanitized(result.stdout + '\n' + result.stderr, token), encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'{label} upload failed; see sanitized log')
        proof = json.loads(result.stdout.splitlines()[-1])
        assert proof['verified'] and proof['verified_files'] == len(manifest['files'])
        assert proof['files'] == manifest['files'] and proof['source_git_commit'] == manifest['source_git_commit']
        known = {x.path: x for x in api.list_bucket_tree(BUCKET, prefix=prefix, recursive=True) if getattr(x, 'xet_hash', None)}
        assert set(known) == {f['remote'] for f in manifest['files']}
        for f in manifest['files']:
            assert known[f['remote']].size == f['bytes'] and known[f['remote']].xet_hash == f['xet_hash']
        proof['independent_local_listing_verified'] = True
        write_json(OUT / (label + '_upload_verified.json'), proof)
        proofs.append(proof)
        write_json(OUT / 'additional_verified.json', proofs)
        print(json.dumps(dict(event='additional_experiment_verified', label=label, verified_files=len(manifest['files']))), flush=True)
    print(json.dumps(dict(phase='complete', additional_experiments=3, verified_files=sum(len(p['files']) for p in proofs))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'upload'))
    prepare() if parser.parse_args().action == 'prepare' else upload()

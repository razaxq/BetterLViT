"""Recover six completed recipe archives using the standard local HF credential cache."""
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, get_token

HERE = Path(__file__).resolve().parent
ROOT = Path('D:/BetterLViT')
OUT = ROOT / 'outputs/hf_recovery_20260910'
DOCS = HERE.parents[2]
KEY = 'C:/Users/dtftn/.ssh/seetacloud_betterlvit_ed25519'
BUCKET = 'razaxq/BetterLViT'
LABELS = ('r1', 'r2', 'c4s2027', 'r2s2027', 'c4s3407', 'r2s3407')
HELPER = DOCS / 'repro_archive/20260908/recipe_execution/upload_hf_recipe_server.py'


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    temporary.replace(path)


def sanitized(value, token):
    return re.sub(r'hf_[A-Za-z0-9]{16,}', '[REDACTED]', value.replace(token, '[REDACTED]'))


def ssh(command, **kwargs):
    return subprocess.run(['ssh', '-i', KEY, '-p', '21465', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                           'root@connect.westb.seetacloud.com', command], text=True, encoding='utf-8',
                          capture_output=True, **kwargs)


def main():
    OUT.mkdir(exist_ok=True)
    token = get_token()
    assert token, 'A credential in the standard HF cache is required'
    api = HfApi(token=token)
    assert api.whoami()['name'] == 'razaxq'
    pending, seen = [], set()
    for label in LABELS:
        folder = ROOT / ('outputs/recipe_20260908' if label in ('r1', 'r2') else 'outputs/recipe_replication_20260909')
        manifest = json.loads((folder / f'{label}_hf_manifest.json').read_text(encoding='utf-8'))
        assert manifest['training_and_validation_returncodes'] == [0, 0]
        assert manifest['classification'] == 'completed_validation_only_80e_pilot'
        assert len(manifest['files']) == 5
        sha = manifest['source_git_commit']
        assert len(sha) == 40 and all(c in '0123456789abcdef' for c in sha)
        assert manifest['bucket_prefix'] == sha[:8] and sha[:8] not in seen
        seen.add(sha[:8])
        for check in manifest['checkpoint_checks']:
            assert check['source_git_commit'] == sha
        expected = {f['remote']: f for f in manifest['files']}
        known = {f.path: f for f in api.list_bucket_tree(BUCKET, prefix=sha[:8], recursive=True) if getattr(f, 'xet_hash', None)}
        assert set(known) <= set(expected), 'Unexpected content under a training commit prefix'
        for name, remote in known.items():
            assert remote.size == expected[name]['bytes'] and remote.xet_hash == expected[name]['xet_hash']
        pending.append(dict(label=label, manifest=manifest, already_verified=len(known), folder=str(folder)))
    plan = dict(bucket=BUCKET, started_unix=time.time(), experiments=pending,
                total_files=30, total_source_bytes=sum(f['bytes'] for x in pending for f in x['manifest']['files']),
                additive_only=True, remote_deletions=False, token_in_artifacts=False)
    write_json(OUT / 'upload_plan.json', plan)
    # Audit only already-completed static artifact files. Do not poll training or Test jobs.
    code = 'MANIFESTS=' + repr([x['manifest'] for x in pending]) + '\nEXPECTED_HELPER=' + repr(hashlib.sha256(HELPER.read_bytes()).hexdigest()) + '\n' + '''
import hashlib,json
from pathlib import Path
import hf_xet
helper=Path('/root/recipe_runs/upload_hf_recipe_server.py')
assert hashlib.sha256(helper.read_bytes()).hexdigest()==EXPECTED_HELPER
verified=[]
for manifest in MANIFESTS:
    path=Path('/root/recipe_runs/hf_staging')/(manifest['bucket_prefix']+'_manifest.json')
    assert json.loads(path.read_text())==manifest
    hashes=hf_xet.hash_files([f['source'] for f in manifest['files']])
    for expected, actual in zip(manifest['files'],hashes):
        assert actual.file_size==expected['bytes'] and actual.hash==expected['xet_hash']
    verified.append(dict(prefix=manifest['bucket_prefix'],files=len(hashes)))
print(json.dumps(dict(source_files_xet_verified=verified,helper_sha256=EXPECTED_HELPER)))
'''
    audit = ssh('ionice -c 3 nice -n 10 /root/autodl-tmp/envs/betterlvit-paper/bin/python -', input=code, timeout=180)
    if audit.returncode:
        raise RuntimeError(sanitized(audit.stderr + audit.stdout[-3000:], token))
    write_json(OUT / 'source_audit.json', json.loads(audit.stdout))
    print(json.dumps(dict(event='preflight_complete', experiments=6, files=30,
                          total_source_bytes=plan['total_source_bytes'], already_verified=sum(x['already_verified'] for x in pending))), flush=True)
    results = []
    for item in pending:
        label, manifest = item['label'], item['manifest']
        started = time.time()
        print(json.dumps(dict(event='upload_started', label=label, prefix=manifest['bucket_prefix'])), flush=True)
        command = ('ionice -c 3 nice -n 10 env PYTHONPATH=/root/autodl-tmp/hf-bucket-client '
                   '/root/autodl-tmp/envs/betterlvit-paper/bin/python /root/recipe_runs/upload_hf_recipe_server.py '
                   '--manifest /root/recipe_runs/hf_staging/' + manifest['bucket_prefix'] + '_manifest.json')
        result = ssh(command, input=token + '\n', timeout=1200)
        (OUT / (label + '_upload.log')).write_text(sanitized(result.stdout + '\n' + result.stderr, token), encoding='utf-8', newline='\n')
        if result.returncode:
            write_json(OUT / 'state.json', dict(phase='failed', label=label, returncode=result.returncode,
                                               completed=results, failed_unix=time.time()))
            raise RuntimeError(f'{label} upload failed; see sanitized log')
        proof = json.loads(result.stdout.splitlines()[-1])
        assert proof['verified'] and proof['verified_files'] == 5 and proof['bucket'] == BUCKET
        assert proof['files'] == manifest['files'] and proof['source_git_commit'] == manifest['source_git_commit']
        # Verify again from this independent local client after remote upload finishes.
        remote = {f.path: f for f in api.list_bucket_tree(BUCKET, prefix=manifest['bucket_prefix'], recursive=True) if getattr(f, 'xet_hash', None)}
        assert set(remote) == {f['remote'] for f in manifest['files']}
        for f in manifest['files']:
            assert remote[f['remote']].size == f['bytes'] and remote[f['remote']].xet_hash == f['xet_hash']
        proof.update(independent_local_listing_verified=True, completed_unix=time.time())
        write_json(OUT / (label + '_upload_verified.json'), proof)
        write_json(Path(item['folder']) / (label + '_hf_upload_verified.json'), proof)
        results.append(dict(label=label, prefix=manifest['bucket_prefix'], source_git_commit=manifest['source_git_commit'],
                            verified_files=5, source_bytes=sum(f['bytes'] for f in manifest['files']), seconds=time.time()-started))
        write_json(OUT / 'state.json', dict(phase='uploading', completed=results))
        print(json.dumps(dict(event='experiment_verified', **results[-1])), flush=True)
    summary = dict(phase='complete', bucket=BUCKET, verified_files=30, experiments=results,
                   total_source_bytes=plan['total_source_bytes'], completed_unix=time.time(),
                   verification='Fresh server source Xet hashes, server Bucket listing and independent local Bucket listing all match.',
                   remote_deletions=False, local_models_deleted=False,
                   scope='Six completed 80-epoch recipe training archives, with their original validation exports. New Test exports are separate.')
    write_json(OUT / 'summary.json', summary)
    write_json(OUT / 'state.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()

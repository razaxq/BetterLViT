"""Additive Bucket upload with source-size/Xet checks; token is supplied on stdin."""
import argparse
import json
import os
from pathlib import Path
import sys
import subprocess
os.environ['HF_TOKEN']=sys.stdin.readline().strip()
assert os.environ['HF_TOKEN']
if Path('/etc/network_turbo').is_file():
    settings=subprocess.run(['bash','-lc','source /etc/network_turbo >/dev/null 2>&1; env -0'],capture_output=True,check=True).stdout
    available=dict(part.decode().split('=',1) for part in settings.split(b'\0') if b'=' in part)
    for key in ('http_proxy','https_proxy','all_proxy','no_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY',
                'SSL_CERT_FILE','SSL_CERT_DIR','REQUESTS_CA_BUNDLE','CURL_CA_BUNDLE','HF_XET_SSL_CERT_FILE'):
        if key in available:os.environ[key]=available[key]
    print('ACADEMIC_ACCELERATION_CONFIGURED',flush=True)
os.environ['HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY']='false'
os.environ['HF_XET_CLIENT_AC_INITIAL_UPLOAD_CONCURRENCY']='1'
os.environ['HF_XET_FIXED_UPLOAD_CONCURRENCY']='1'
os.environ['HF_XET_DEDUPLICATION_GLOBAL_DEDUP_QUERY_ENABLED']='false'
os.environ['HF_XET_CACHE']='/root/recipe_runs/hf_upload_cache'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
from huggingface_hub import HfApi

p=argparse.ArgumentParser()
p.add_argument('--manifest',type=Path,required=True)
args=p.parse_args()
manifest=json.loads(args.manifest.read_text())
assert manifest['classification']=='completed_validation_only_80e_independent_seed_replication'
assert manifest['training_and_validation_returncodes']==[0,0] and not manifest['test_split_accessed']
assert len(manifest['source_git_commit'])==40 and manifest['bucket_prefix']==manifest['source_git_commit'][:8]
api=HfApi()
bucket='razaxq/BetterLViT'
prefix=manifest['bucket_prefix']
expected={item['remote'] for item in manifest['files']}
known={x.path:x for x in api.list_bucket_tree(bucket,prefix=prefix,recursive=True) if getattr(x,'xet_hash',None)}
assert set(known)<=expected, 'Unexpected paths in this commit prefix'
for item in manifest['files']:
    if item['remote'] in known:
        got=known[item['remote']]
        assert got.size==item['bytes'] and got.xet_hash==item['xet_hash']
        continue
    assert Path(item['source']).stat().st_size==item['bytes']
    print('UPLOAD_BEGIN',item['remote'],flush=True)
    same=next((x for x in known.values() if x.xet_hash==item['xet_hash']),None)
    if item.get('copy_from'):
        parent_prefix=item['copy_from'].split('/')[0]
        parent={x.path:x for x in api.list_bucket_tree(bucket,prefix=parent_prefix,recursive=True) if getattr(x,'xet_hash',None)}
        original=parent[item['copy_from']]
        assert original.size==item['bytes'] and original.xet_hash==item['xet_hash']
        api.batch_bucket_files(bucket,copy=[('bucket',bucket,item['xet_hash'],item['remote'])])
    elif same:
        api.batch_bucket_files(bucket,copy=[('bucket',bucket,item['xet_hash'],item['remote'])])
    else:
        api.batch_bucket_files(bucket,add=[(str(item['source']),item['remote'])])
    known={x.path:x for x in api.list_bucket_tree(bucket,prefix=prefix,recursive=True) if getattr(x,'xet_hash',None)}
    got=known[item['remote']]
    assert got.size==item['bytes'] and got.xet_hash==item['xet_hash']
    print('HF_VERIFIED',item['remote'],flush=True)
assert set(known)==expected
manifest.update(verified=True,verified_files=len(expected),bucket=bucket)
proof=args.manifest.with_name(prefix+'_upload_verified.json')
proof.write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest),flush=True)

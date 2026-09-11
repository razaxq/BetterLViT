"""Archive only completed, independently verified F artifacts under its source SHA."""
import json
import os
from pathlib import Path
import subprocess
import time
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
os.environ['HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY']='false'
os.environ['HF_XET_FIXED_UPLOAD_CONCURRENCY']='1'
import hf_xet
from huggingface_hub import HfApi,get_token
from analysis import digest,write_json
HERE=Path(__file__).resolve().parent;DOCS=next(p for p in HERE.parents if (p/'.git').exists())
runtime=json.loads((HERE/'results/runtime.json').read_text());assert runtime['phase']=='complete'
proof=json.loads((HERE/'results/independent_verification.json').read_text());assert proof['verified']
ck=json.loads((HERE/'checkpoint_verified.json').read_text());assert ck['verified'] and ck['heads']==20 and ck['resume_optimizer_load_verified']
sha=runtime['source_git_commit'];prefix=sha[:8];bucket='razaxq/BetterLViT'
assert sha==json.loads((HERE/'deployment.json').read_text())['source_git_commit']==proof['source_git_commit']==ck['source_git_commit']
assert subprocess.check_output(['git','rev-parse','pilot-local-refinement-f2-20260912'],cwd=DOCS,text=True).strip()==sha
for name,meta in runtime['artifacts'].items():assert digest(HERE/'results'/name)==meta['sha256']
stage=Path('D:/BetterLViT/.codex_tmp/hf_local_refinement_'+prefix);stage.mkdir(exist_ok=True)
archive=stage/'source.tar.gz'
subprocess.run(['git','archive','--format=tar.gz','--output',str(archive),sha,HERE.relative_to(DOCS).as_posix()],cwd=DOCS,check=True)
mapping=[(archive,prefix+'/source.tar.gz')]
mapping += [(HERE/'results'/name,prefix+'/run/'+name) for name in sorted(runtime['artifacts'])]
mapping += [(HERE/'results'/name,prefix+'/run/'+name) for name in ('runtime.json','run.log','summary_verified.json','independent_verification.json')]
mapping += [(HERE/'preflight'/name,prefix+'/preflight/'+name) for name in ('runtime.json','cache_manifest.json','synthetic_checks.json','benchmark.json')]
mapping += [(HERE/name,prefix+'/'+name) for name in ('manifest.json','split.json','PROTOCOL.md','REPORT.md','checkpoint_verified.json','state.json')]
hashes=hf_xet.hash_files([str(p) for p,_ in mapping])
files=[dict(local=str(path),remote=remote,bytes=h.file_size,xet_hash=h.hash,sha256=digest(path)) for (path,remote),h in zip(mapping,hashes)]
manifest=dict(source_git_commit=sha,classification='completed_frozen_R2_Train_head_crossfit_pilot',
    formal_test_result=False,official_validation_accessed=False,test_split_accessed=False,files=files,
    folds=5,heads=20,steps_per_head=2048,checkpoint_load_verified=True,additive_only=True)
write_json(HERE/'hf_manifest.json',manifest)
token=get_token();assert token;api=HfApi(token=token);assert api.whoami()['name']=='razaxq'
expected={f['remote']:f for f in files}
known={f.path:f for f in api.list_bucket_tree(bucket,prefix=prefix,recursive=True) if getattr(f,'xet_hash',None)}
assert set(known)<=set(expected)
for name,value in known.items():assert value.size==expected[name]['bytes'] and value.xet_hash==expected[name]['xet_hash']
missing=[(f['local'],f['remote']) for f in files if f['remote'] not in known]
print(json.dumps(dict(event='upload_begin',files=len(files),missing=len(missing),bytes=sum(f['bytes'] for f in files))),flush=True)
if missing:api.batch_bucket_files(bucket,add=missing)
known={f.path:f for f in api.list_bucket_tree(bucket,prefix=prefix,recursive=True) if getattr(f,'xet_hash',None)}
assert set(known)==set(expected)
for name,value in known.items():assert value.size==expected[name]['bytes'] and value.xet_hash==expected[name]['xet_hash']
result=dict(verified=True,source_git_commit=sha,bucket=bucket,bucket_prefix=prefix,classification=manifest['classification'],
    files_verified=len(files),logical_bytes=sum(f['bytes'] for f in files),all_sizes_and_xet_hashes_match=True,
    additive_only=True,remote_deletions=False,formal_test_result=False,test_split_accessed=False,completed_unix=time.time())
write_json(HERE/'hf_upload_verified.json',result);print(json.dumps(result))

"""Additive archive of a completed no-update Train diagnostic, not a Test run."""
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
HERE=Path(__file__).resolve().parent
DOCS=next(p for p in HERE.parents if (p/'.git').exists())
BUCKET='razaxq/BetterLViT'
proof=json.loads((HERE/'results/independent_verification.json').read_text());assert proof['verified']
runtime=json.loads((HERE/'results/runtime.json').read_text());assert runtime['phase']=='complete' and runtime['train_updates']==0
sha=runtime['source_git_commit'];prefix=sha[:8]
assert sha=='66db167b48e4e0eeb91738af90df6abb6a78ddad' and proof['source_git_commit']==sha
for name,meta in runtime['artifacts'].items():assert digest(HERE/'results'/name)==meta['sha256']
assert subprocess.check_output(['git','rev-parse','diagnostic-text-iou-e2-20260912'],cwd=DOCS,text=True).strip()==sha
stage=Path('D:/BetterLViT/.codex_tmp/hf_text_iou_'+prefix);stage.mkdir(exist_ok=True)
archive=stage/'source.tar.gz'
subprocess.run(['git','archive','--format=tar.gz','--output',str(archive),sha,HERE.relative_to(DOCS).as_posix()],cwd=DOCS,check=True)
mapping=[(archive,prefix+'/source.tar.gz')]
mapping += [(HERE/'results'/name,prefix+'/run/'+name) for name in sorted(runtime['artifacts'])]
mapping += [(HERE/'results'/name,prefix+'/run/'+name) for name in ('runtime.json','run.log','independent_verification.json','summary_verified.json')]
mapping += [(HERE/name,prefix+'/'+name) for name in ('manifest.json','PROTOCOL.md','selection.json','REPORT.md','NEXT_PLAN.md','report_completed.py','closure.json')]
hashes=hf_xet.hash_files([str(p) for p,_ in mapping])
files=[dict(local=str(path),remote=remote,bytes=h.file_size,xet_hash=h.hash,sha256=digest(path)) for (path,remote),h in zip(mapping,hashes)]
manifest=dict(source_git_commit=sha,bucket_prefix=prefix,classification='completed_train_only_readonly_feasibility_audit',
    formal_test_result=False,train_updates=0,official_validation_accessed=False,test_split_accessed=False,files=files,
    dependency_b_prefix='49905dbb',dependency_d_prefix='c7080ea8',new_checkpoint=False,additive_only=True)
write_json(HERE/'hf_manifest.json',manifest)
token=get_token();assert token
api=HfApi(token=token);assert api.whoami()['name']=='razaxq'
expected={f['remote']:f for f in files}
known={f.path:f for f in api.list_bucket_tree(BUCKET,prefix=prefix,recursive=True) if getattr(f,'xet_hash',None)}
assert set(known)<=set(expected)
for name,value in known.items():assert value.size==expected[name]['bytes'] and value.xet_hash==expected[name]['xet_hash']
missing=[(f['local'],f['remote']) for f in files if f['remote'] not in known]
print(json.dumps(dict(event='upload_begin',files=len(files),missing=len(missing),bytes=sum(f['bytes'] for f in files))),flush=True)
if missing:api.batch_bucket_files(BUCKET,add=missing)
known={f.path:f for f in api.list_bucket_tree(BUCKET,prefix=prefix,recursive=True) if getattr(f,'xet_hash',None)}
assert set(known)==set(expected)
for name,value in known.items():assert value.size==expected[name]['bytes'] and value.xet_hash==expected[name]['xet_hash']
write_json(HERE/'hf_upload_verified.json',dict(verified=True,source_git_commit=sha,bucket=BUCKET,bucket_prefix=prefix,
    classification=manifest['classification'],files_verified=len(files),logical_bytes=sum(f['bytes'] for f in files),
    all_sizes_and_xet_hashes_match=True,additive_only=True,remote_deletions=False,train_updates=0,
    formal_test_result=False,test_split_accessed=False,completed_unix=time.time()))
print((HERE/'hf_upload_verified.json').read_text())

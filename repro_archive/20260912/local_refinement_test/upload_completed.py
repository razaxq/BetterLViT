"""Add verified Test exports, exact evaluator source and frozen head files to HF."""
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
HERE=Path(__file__).resolve().parent;DOCS=HERE.parents[2];F=HERE.parent/'local_refinement_screen'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
rt=read(HERE/'results/runtime.json');summary=read(HERE/'summary.json');a=read(HERE/'authorization.json')
assert rt['phase']=='complete' and summary['verified'] and summary['samples']==2113
assert read(HERE/'source_verified.json')['verified']
sha=rt['evaluation_source_git_commit'];assert sha==summary['evaluation_source_git_commit']
prefix=sha[:8];bucket='razaxq/BetterLViT'
stage=Path('D:/BetterLViT/.codex_tmp/hf_local_refinement_test_'+prefix);stage.mkdir(exist_ok=True)
archive=stage/'source.tar.gz'
subprocess.run(['git','archive','--format=tar.gz','--output',str(archive),sha,HERE.relative_to(DOCS).as_posix()],cwd=DOCS,check=True)
mapping=[(archive,prefix+'/source.tar.gz')]
for name,meta in rt['artifacts'].items():
    path=HERE/'results'/name;assert digest(path)==meta['sha256'];mapping.append((path,prefix+'/test/'+name))
mapping += [(HERE/'results'/name,prefix+'/test/'+name) for name in ('runtime.json','run.log')]
mapping += [(HERE/name,prefix+'/'+name) for name in ('PROTOCOL.md','authorization.json','launch.json','state.json','summary.json','REPORT.md','source_verified.json')]
for name,meta in a['head_files'].items():
    path=F/'results'/name;assert digest(path)==meta['sha256'];mapping.append((path,prefix+'/heads/'+name))
mapping.append((F/'checkpoint_verified.json',prefix+'/heads/checkpoint_verified.json'))
hashes=hf_xet.hash_files([str(p) for p,_ in mapping])
files=[dict(local=str(p),remote=remote,bytes=h.file_size,xet_hash=h.hash,sha256=digest(p)) for (p,remote),h in zip(mapping,hashes)]
manifest=dict(evaluation_source_git_commit=sha,f_source_git_commit=a['f_source_git_commit'],
    classification='user_requested_fixed_F_heads_Test_evaluation_after_failed_Train_screen',
    test_split_accessed=True,no_new_training=True,no_test_selection=True,files=files)
write_json(HERE/'hf_manifest.json',manifest)
api=HfApi(token=get_token());assert api.whoami()['name']=='razaxq'
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
proof=dict(verified=True,evaluation_source_git_commit=sha,f_source_git_commit=a['f_source_git_commit'],bucket=bucket,
    bucket_prefix=prefix,files_verified=len(files),logical_bytes=sum(f['bytes'] for f in files),
    all_sizes_and_xet_hashes_match=True,additive_only=True,remote_deletions=False,test_split_accessed=True,completed_unix=time.time())
write_json(HERE/'hf_upload_verified.json',proof);print(json.dumps(proof))

"""Additively back up completed inference evidence, without duplicating weights."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

import hf_xet
from huggingface_hub import HfApi

HERE=Path(__file__).resolve().parent
DOCS=HERE.parents[2]
proof=json.loads((HERE/'provenance_verified.json').read_text());assert proof['verified']
assert json.loads((HERE/'test/summary.json').read_text())['verified']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=DOCS,text=True).strip()
docsha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=DOCS,text=True).strip()
source=proof['evaluation_source_git_commit'];prefix=source[:8];bucket='razaxq/BetterLViT'
output=Path('D:/BetterLViT/outputs')/('rapid_iou_'+prefix);output.mkdir(exist_ok=True)
archive=output/'inference_evidence.tar.gz'
assert not archive.exists(), 'Preserve the first completed evidence package'
files=sorted(p for p in HERE.rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json','.log')
             and not p.name.startswith('hf_'))
entries={p.relative_to(HERE).as_posix():dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}
with tarfile.open(archive,'w:gz') as tar:
    for p in files:tar.add(p,arcname=p.relative_to(HERE).as_posix())
manifest=output/'manifest.json'
data=dict(classification='completed_single_model_inference_threshold_evaluation',
    evaluation_source_git_commit=source,documentation_source_git_commit=docsha,
    original_checkpoint_source_git_commit='9eca26de5b301099805530edbf5a1a8718bea662',
    checkpoint_sha256=proof['checkpoint_sha256'],no_new_weights=True,test_split_accessed=True,
    threshold_selected_on_validation=.54,files=entries)
manifest.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8',newline='\n')
paths=[archive,manifest];hashes=hf_xet.hash_files([str(p) for p in paths])
expected={prefix+'/'+p.name:dict(bytes=h.file_size,xet_hash=h.hash) for p,h in zip(paths,hashes)}
api=HfApi();assert api.whoami()['name']=='razaxq'
known={f.path:f for f in api.list_bucket_tree(bucket,prefix=prefix,recursive=True) if getattr(f,'xet_hash',None)}
assert set(known)<=set(expected), 'Unexpected content under the evaluation source prefix'
for name,f in known.items():assert f.size==expected[name]['bytes'] and f.xet_hash==expected[name]['xet_hash']
missing=[(str(p),prefix+'/'+p.name) for p in paths if prefix+'/'+p.name not in known]
if missing:api.batch_bucket_files(bucket,add=missing)
fresh={f.path:f for f in api.list_bucket_tree(bucket,prefix=prefix,recursive=True) if getattr(f,'xet_hash',None)}
assert set(fresh)==set(expected)
for name,f in fresh.items():assert f.size==expected[name]['bytes'] and f.xet_hash==expected[name]['xet_hash']
result=dict(verified=True,bucket=bucket,prefix=prefix,files=expected,
    logical_bytes=sum(f['bytes'] for f in expected.values()),archived_file_count=len(entries),
    additive_only=True,new_weights_uploaded=False,documentation_source_git_commit=docsha)
(HERE/'hf_upload_verified.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(result))

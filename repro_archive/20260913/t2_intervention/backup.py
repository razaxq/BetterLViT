"""Additive archival of completed diagnostic evidence, no checkpoint copies."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import hf_xet
from huggingface_hub import HfApi

HERE=Path(__file__).resolve().parent
DOCS=HERE.parents[2]
assert json.loads((HERE/'analysis.json').read_text())['verified']
assert json.loads((HERE/'binding_analysis.json').read_text())['verified']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=DOCS,text=True).strip()
sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=DOCS,text=True).strip()
source=json.loads((HERE/'binding_result.json').read_text())['analysis_git_commit']
prefix=source[:8];bucket='razaxq/BetterLViT'
output=Path('D:/BetterLViT/outputs')/('t2_intervention_'+prefix)
output.mkdir(exist_ok=True)
archive=output/'diagnostic_evidence.tar.gz'
assert not archive.exists(),'Preserve original evidence package'
files=sorted(p for p in HERE.rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json','.log','.txt') and not p.name.startswith('hf_'))
files += [HERE.parent/'text_decoder_controls'/n for n in ('analyze.py','remote_ops.py','sources.json','t2_results/validation.json')]
entries={p.relative_to(DOCS).as_posix():dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}
with tarfile.open(archive,'w:gz') as tar:
    for p in files:tar.add(p,arcname=p.relative_to(DOCS).as_posix())
manifest=output/'manifest.json'
manifest.write_text(json.dumps(dict(classification='completed_validation_interventions_and_train_gradient_diagnostics',
    documentation_source_git_commit=sha,analysis_source_git_commit=source,
    checkpoint_source_git_commit='488ef093de80df71ee77741a8c7ee7b938c7d6b5',
    second_innovation_established=False,training_performed=False,test_split_accessed=False,no_new_weights=True,files=entries),indent=2)+'\n')
paths=[archive,manifest];hashes=hf_xet.hash_files([str(p) for p in paths])
expected={prefix+'/'+p.name:dict(bytes=h.file_size,xet_hash=h.hash) for p,h in zip(paths,hashes)}
api=HfApi();assert api.whoami()['name']=='razaxq'
known={f.path:f for f in api.list_bucket_tree(bucket,prefix=prefix,recursive=True) if getattr(f,'xet_hash',None)}
assert set(known)<=set(expected)
for n,f in known.items():assert f.size==expected[n]['bytes'] and f.xet_hash==expected[n]['xet_hash']
missing=[(str(p),prefix+'/'+p.name) for p in paths if prefix+'/'+p.name not in known]
if missing:api.batch_bucket_files(bucket,add=missing)
fresh={f.path:f for f in api.list_bucket_tree(bucket,prefix=prefix,recursive=True) if getattr(f,'xet_hash',None)}
assert set(fresh)==set(expected)
for n,f in fresh.items():assert f.size==expected[n]['bytes'] and f.xet_hash==expected[n]['xet_hash']
proof=dict(verified=True,bucket=bucket,prefix=prefix,files=expected,archived_file_count=len(files),
    additive_only=True,new_weights_uploaded=False,documentation_source_git_commit=sha)
(HERE/'hf_upload_verified.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof))

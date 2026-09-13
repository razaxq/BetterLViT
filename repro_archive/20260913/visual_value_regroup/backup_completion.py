"""Additively archive final decisions and reproducibility evidence; no new weights."""
import hashlib,json,subprocess,tarfile
from pathlib import Path
import hf_xet
from huggingface_hub import HfApi
from remote_ops import HERE,DOCS,read,save

decision=read(HERE/'discovery_decision.json')
assert (HERE/'RESULTS_ZH.md').is_file()
assert read(HERE/'automation_deleted_verified.json')['verified']
for label in ('m1','m2'):
    for name in ('independent_verification.json','hf_upload_verified.json','download_xet_verified.json','github_archive_verified.json'):
        assert read(HERE/(label+'_results')/name)['verified']
    assert read(HERE/(label+'_state.json'))['last_phase']=='complete'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=DOCS,text=True).strip()
sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=DOCS,text=True).strip()
prefix=sha[:8];bucket='razaxq/BetterLViT'
output=Path('D:/BetterLViT/outputs')/('visual_regroup_completion_'+prefix)
output.mkdir(exist_ok=True);archive=output/'discovery_evidence.tar.gz';manifest_path=output/'manifest.json'
files=sorted(p for p in HERE.rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json','.log','.txt'))
files += [DOCS/'repro_archive/20260908/recipe_execution/r2_results/validation.json']
entries={p.relative_to(DOCS).as_posix():dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}
manifest=dict(classification='completed_visual_value_regrouping_discovery',documentation_source_git_commit=sha,
    training_sources=read(HERE/'sources.json'),discovery_gate_passed=decision['discovery_gate_passed'],
    second_innovation_established=False,test_split_accessed=False,new_weights_uploaded=False,files=entries)
if archive.exists():
    assert read(manifest_path)==manifest,'Existing evidence package is immutable'
else:
    with tarfile.open(archive,'w:gz') as tar:
        for p in files:tar.add(p,arcname=p.relative_to(DOCS).as_posix())
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
paths=[archive,manifest_path];hashes=hf_xet.hash_files([str(p) for p in paths])
expected={prefix+'/'+p.name:dict(bytes=h.file_size,xet_hash=h.hash) for p,h in zip(paths,hashes)}
api=HfApi();assert api.whoami()['name']=='razaxq'
known={f.path:f for f in api.list_bucket_tree(bucket,prefix=prefix,recursive=True) if getattr(f,'xet_hash',None)}
assert set(known)<=set(expected)
for name,value in known.items():assert value.size==expected[name]['bytes'] and value.xet_hash==expected[name]['xet_hash']
missing=[(str(p),prefix+'/'+p.name) for p in paths if prefix+'/'+p.name not in known]
if missing:api.batch_bucket_files(bucket,add=missing)
fresh={f.path:f for f in api.list_bucket_tree(bucket,prefix=prefix,recursive=True) if getattr(f,'xet_hash',None)}
assert set(fresh)==set(expected)
for name,value in fresh.items():assert value.size==expected[name]['bytes'] and value.xet_hash==expected[name]['xet_hash']
proof=dict(verified=True,bucket=bucket,prefix=prefix,files=expected,archived_file_count=len(files),
    logical_bytes=sum(x['bytes'] for x in expected.values()),additive_only=True,new_weights_uploaded=False,
    documentation_source_git_commit=sha,discovery_gate_passed=decision['discovery_gate_passed'])
save('completion_backup_verified.json',proof);print(json.dumps(proof))

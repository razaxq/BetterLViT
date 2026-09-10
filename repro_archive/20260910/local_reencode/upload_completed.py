"""Add independently verified local re-encoding artifacts to the user's HF Bucket."""
import json
import subprocess
from pathlib import Path
import hf_xet
from huggingface_hub import HfApi
from huggingface_hub.utils import disable_progress_bars
from analysis import digest, write_json

HERE=Path(__file__).resolve().parent
DOCS=HERE.parents[2]
OUT=Path('D:/BetterLViT/outputs/local_reencode_20260910')
BUCKET='razaxq/BetterLViT'
SHA='c749b72db327cd777f63a4b6da4583b301456538'
PREFIX=SHA[:8]+'/local_reencode_v1'


def main():
    disable_progress_bars()
    state=json.loads((OUT/'state.json').read_text())
    proof=json.loads((HERE/'results/independent_verification.json').read_text())
    assert state['phase']=='complete' and proof['verified']
    assert state['source_git_commit']==SHA
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=DOCS,text=True).strip(),'Commit archive first'
    api=HfApi();assert api.whoami()['name']=='razaxq'
    known={x.path:x for x in api.list_bucket_tree(BUCKET,prefix=PREFIX,recursive=True) if getattr(x,'xet_hash',None)}
    mapping=[(p,PREFIX+'/results/'+p.name) for p in sorted((HERE/'results').iterdir()) if p.is_file()]
    mapping.extend((p,PREFIX+'/execution/'+p.name) for p in sorted((HERE/'execution').iterdir())
                   if p.suffix=='.json' and p.name!='hf_upload_verified.json')
    frozen=OUT/'frozen_upload_source';frozen.mkdir(exist_ok=True)
    for name in ('run.py','model.py','analysis.py','control.py','check_analysis.py','manifest.json','PLAN.md','r2_validation.json'):
        data=subprocess.check_output(['git','show',SHA+':'+(HERE/name).relative_to(DOCS).as_posix()],cwd=DOCS)
        path=frozen/name;path.write_bytes(data);mapping.append((path,PREFIX+'/source/'+name))
    hashes=hf_xet.hash_files([str(p) for p,_ in mapping]);pending=[]
    for (path,remote),info in zip(mapping,hashes):
        if remote in known:
            assert known[remote].size==info.file_size and known[remote].xet_hash==info.hash,'Existing object differs'
        else:pending.append((path,remote))
    if pending:api.batch_bucket_files(BUCKET,add=pending)
    verified={x.path:x for x in HfApi().list_bucket_tree(BUCKET,prefix=PREFIX,recursive=True) if getattr(x,'xet_hash',None)}
    rows=[]
    for (path,remote),info in zip(mapping,hashes):
        assert verified[remote].size==info.file_size and verified[remote].xet_hash==info.hash
        rows.append(dict(remote=remote,bytes=info.file_size,xet_hash=info.hash,sha256=digest(path)))
    result=dict(verified=True,bucket=BUCKET,prefix=PREFIX,source_git_commit=SHA,additive_only=True,
                files=rows,verified_files=len(rows),logical_bytes=sum(r['bytes'] for r in rows))
    write_json(OUT/'hf_upload_verified.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='files'}))


if __name__=='__main__':main()

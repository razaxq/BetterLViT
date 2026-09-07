"""Produce a deletion plan only for byte-identical HF-backed Last checkpoints."""
import json
import os
from pathlib import Path
import shutil
import sys
os.environ['HF_TOKEN']=sys.stdin.readline().strip()
import hf_xet
from huggingface_hub import HfApi

root=Path('/root/autodl-tmp').resolve()
out=Path('/root/maintenance_20260907')
api=HfApi()
remote=[x for x in api.list_bucket_tree('razaxq/BetterLViT',recursive=True) if getattr(x,'xet_hash',None)]
lasts=[x for x in remote if x.path.endswith('/models/last_model-BetterLViT.pth.tar')]
by_size={}
by_path={x.path:x for x in remote}
for x in lasts: by_size.setdefault(x.size,[]).append(x)
plan=[]; skipped=[]
for last in sorted(root.glob('BetterLViT*/Covid19/BetterLViT/*/*/models/last_model-BetterLViT.pth.tar')):
    best=last.with_name('best_model-BetterLViT.pth.tar')
    candidates=by_size.get(last.stat().st_size,[])
    if not candidates or not best.is_file():
        skipped.append(str(last)); continue
    info=hf_xet.hash_files([str(last)])[0]
    matches=[x for x in candidates if x.xet_hash==info.hash]
    if not matches:
        skipped.append(str(last)); continue
    match=matches[0]
    remote_best=by_path.get(match.path.replace('/last_model-','/best_model-'))
    if remote_best is None or remote_best.size!=best.stat().st_size:
        skipped.append(str(last)); continue
    binfo=hf_xet.hash_files([str(best)])[0]
    if remote_best.xet_hash!=binfo.hash:
        skipped.append(str(last)); continue
    stat=last.stat()
    plan.append({'path':str(last.resolve()),'best_preserved':str(best.resolve()),'bytes':stat.st_size,
                 'mtime_ns':stat.st_mtime_ns,'inode':stat.st_ino,'nlink':stat.st_nlink,
                 'remote_last':match.path,'remote_best':remote_best.path,
                 'last_xet_hash':info.hash,'best_xet_hash':binfo.hash})
    print('VERIFIED_CLEANUP_CANDIDATE',match.path,stat.st_size,flush=True)
usage=shutil.disk_usage(root)
result={'root':str(root),'disk_before':dict(zip(('total','used','free'),usage)),
        'verified_candidates':plan,'skipped_unverified':skipped,
        'logical_bytes':sum(x['bytes'] for x in plan),
        'single_link_bytes':sum(x['bytes'] for x in plan if x['nlink']==1)}
(out/'cleanup_plan.json').write_text(json.dumps(result,indent=2))
print('PLAN_COMPLETE',len(plan),result['logical_bytes'],'BYTES','SKIPPED',len(skipped),flush=True)

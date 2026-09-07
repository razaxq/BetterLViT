"""Archive completed, Test-evaluated exploratory runs with Xet verification."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

prepare_only='--prepare-only' in sys.argv
if not prepare_only:
    os.environ['HF_TOKEN'] = sys.stdin.readline().strip()
if not prepare_only and not os.environ['HF_TOKEN']:
    raise RuntimeError('Missing token on stdin')
import torch
import hf_xet
from huggingface_hub import HfApi

OUT=Path('/root/maintenance_20260907')
OUT.mkdir(exist_ok=True)
BUCKET='razaxq/BetterLViT'
api=HfApi()
arms=[
 ('c4','/root/race_pe_test_20260907','/root/race_pe_runs/c4_p9_20260906'),
 ('p9','/root/race_pe_test_20260907','/root/race_pe_runs/c4_p9_20260906'),
 ('c8','/root/race_pe_v2_test_20260907','/root/race_pe_v2_runs/c8_p10_20260907'),
 ('p10','/root/race_pe_v2_test_20260907','/root/race_pe_v2_runs/c8_p10_20260907')]
proof=[]
for arm,testdir,valdir in arms:
    assert (Path(testdir)/'status').read_text().strip()=='complete'
    assert (Path(valdir)/'chain.status').read_text().strip()=='complete'
    testpath=Path(testdir)/f'{arm}_test.json'
    result=json.loads(testpath.read_text())
    val=json.loads((Path(valdir)/f'{arm}_validation.json').read_text())
    commit=result['checkpoint_git_commit']
    assert commit==val['checkpoint_git_commit'] and len(commit)==40
    assert result['samples']==2113 and result['split']=='test' and result['threshold']==0.5
    checkpoint=Path(result['checkpoint'])
    session=checkpoint.parent.parent
    best=checkpoint
    last=checkpoint.parent/'last_model-BetterLViT.pth.tar'
    checks=[]
    for model in (best,last):
        data=torch.load(model,map_location='cpu',weights_only=False)
        assert data['source_git_commit']==commit
        assert data['state_dict']
        checks.append({k:data.get(k) for k in ('epoch','best_epoch','source_git_commit','architecture_version')})
        del data
    logs=list(session.glob('*.log'))
    events=list((session/'tensorboard_logs').glob('events.out*'))
    assert len(logs)==1 and len(events)==1
    mapping={logs[0].name:logs[0],f'{arm}_test_evaluation.json':testpath,
             'models/'+best.name:best,'models/'+last.name:last,
             'tensorboard_logs/'+events[0].name:events[0]}
    prefix=commit[:8]
    stage=OUT/'upload_staging'/prefix
    for name,src in mapping.items():
        dst=stage/name; dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():
            assert os.path.samefile(src,dst)
        else:
            os.link(src,dst)
    if prepare_only:
        hashes=hf_xet.hash_files([str(x) for x in mapping.values()])
        proof.append({'arm':arm,'source_git_commit':commit,'checkpoint_checks':checks,
                      'files':[{'source':str(src),'remote':f'{prefix}/{name}','bytes':info.file_size,
                                'xet_hash':info.hash} for (name,src),info in zip(mapping.items(),hashes)]})
        (OUT/'transfer_manifest.json').write_text(json.dumps(proof,indent=2))
        print('PREPARED',arm,len(mapping),flush=True)
        continue
    before=list(api.list_bucket_tree(BUCKET,prefix=prefix,recursive=True))
    existing={x.path for x in before if getattr(x,'xet_hash',None)}
    expected={f'{prefix}/{name}' for name in mapping}
    assert existing<=expected, (prefix,'unexpected existing paths')
    print('UPLOAD_BEGIN',arm,commit,sum(x.stat().st_size for x in mapping.values()),flush=True)
    # Serialize uploads: identical Best/Last files can share Xet data without
    # simultaneous cleaning of the same large content in a single sync batch.
    hashes=hf_xet.hash_files([str(x) for x in mapping.values()])
    known={x.path:x for x in before if getattr(x,'xet_hash',None)}
    uploaded_hashes=set()
    for (name,src),info in zip(mapping.items(),hashes):
        destination=f'{prefix}/{name}'
        if destination in known:
            assert known[destination].xet_hash==info.hash and known[destination].size==info.file_size
        elif info.hash in uploaded_hashes:
            api.batch_bucket_files(BUCKET,copy=[('bucket',BUCKET,info.hash,destination)])
        else:
            print('FILE_BEGIN',arm,name,flush=True)
            api.batch_bucket_files(BUCKET,add=[(str(src),destination)])
        uploaded_hashes.add(info.hash)
        print('FILE_STORED',arm,name,flush=True)
    remote={x.path:x for x in api.list_bucket_tree(BUCKET,prefix=prefix,recursive=True)
            if getattr(x,'xet_hash',None)}
    assert set(remote)==expected
    files=[]
    for (name,src),info in zip(mapping.items(),hashes):
        target=remote[f'{prefix}/{name}']
        assert target.size==src.stat().st_size==info.file_size
        assert target.xet_hash==info.hash
        files.append({'source':str(src),'remote':target.path,'bytes':target.size,'xet_hash':info.hash})
    proof.append({'arm':arm,'source_git_commit':commit,'classification':'completed_exploratory_80e',
                  'test_macro_iou':result['macro_iou'],'test_macro_dice':result['macro_dice'],
                  'checkpoint_checks':checks,'files':files,'verified':True})
    (OUT/'new_runs_upload_verified.json').write_text(json.dumps(proof,indent=2))
    print('VERIFIED',arm,len(files),'OF',len(mapping),flush=True)
print('ALL_PREPARED' if prepare_only else 'ALL_VERIFIED',len(proof),'RUNS',sum(len(x['files']) for x in proof),'FILES',flush=True)

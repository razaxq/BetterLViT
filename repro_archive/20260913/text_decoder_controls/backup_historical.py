"""Preserve completed historical pilot checkpoints without inventing evaluations.

These are checkpoint/storage archives, NOT newly certified formal results.
Broader user authorization covers historical training artifacts and source/docs.
"""
import base64
import json
import re
import subprocess
import time
from huggingface_hub import HfApi,get_token
from remote_ops import HERE,KEY,HOST,PYTHON,read,remote,save
items=read(HERE/'storage_candidates.json')[1:5]
helper=(HERE.parents[1]/'20260912/rs1_iou_replication/upload_verified_manifest.py').read_text()
helper=helper.replace("assert manifest['classification']=='completed_validation_only_80e_independent_seed_replication'",
    "assert manifest['classification']=='completed_historical_pilot_checkpoint_archive'")
helper=helper.replace("assert manifest['training_and_validation_returncodes']==[0,0] and not manifest['test_split_accessed']",
    "assert manifest['checkpoint_completed_epochs']==manifest['configured_epochs']\nassert not manifest['new_evaluation_performed']")
(HERE/'historical_checkpoint_uploader.py').write_text(helper,encoding='utf-8',newline='\n')
token=get_token();assert token
api=HfApi(token=token);assert api.whoami()['name']=='razaxq'
for item in items:
    sha=item['source_git_commit'];prefix=sha[:8]
    manifest=remote('ITEM='+repr(item)+'\nHELPER='+repr(base64.b64encode(helper.encode()).decode())+'\n'+'''
import base64,json,subprocess
from pathlib import Path
import hf_xet,torch
sha=ITEM['source_git_commit'];prefix=sha[:8]
best=Path(ITEM['best']);last=Path(ITEM['last']);session=best.parent.parent
repo=next(p for p in best.parents if (p/'.git').exists())
stage=Path('/root/autodl-tmp/text_decoder_runs/historical_backup')/prefix
stage.mkdir(parents=True,exist_ok=True)
(stage/'upload.py').write_bytes(base64.b64decode(HELPER))
checks=[]
for p in (best,last):
    c=torch.load(p,map_location='cpu',weights_only=True)
    assert c['source_git_commit']==sha
    if p==last:assert c['epoch']+1==c['epochs']==len(c['epoch_history'])
    checks.append(dict(path=str(p),source_git_commit=sha,epoch=c['epoch'],epochs=c['epochs'],
        history_rows=len(c['epoch_history']),best_epoch=c['best_epoch']))
    del c
metadata=dict(classification='completed_historical_pilot_checkpoint_archive',checkpoint_checks=checks,
    new_evaluation_performed=False,formal_evaluation_status='not_recertified_by_storage_backup',
    purpose='Preserve completed pilot source and checkpoints before removing local inactive Last copy')
(stage/'checkpoint_archive.json').write_text(json.dumps(metadata,indent=2)+'\\n')
archive=stage/'source.tar.gz'
subprocess.run(['git','archive','--format=tar.gz','--output',str(archive),sha],cwd=repo,check=True)
mapping=[(best,prefix+'/models/'+best.name),(last,prefix+'/models/'+last.name),
         (archive,prefix+'/source.tar.gz'),(stage/'checkpoint_archive.json',prefix+'/checkpoint_archive.json')]
mapping.extend((p,prefix+'/'+p.name) for p in session.glob('*.log'))
mapping.extend((p,prefix+'/tensorboard/'+p.name) for p in session.rglob('events.out.tfevents.*'))
hashes=hf_xet.hash_files([str(p) for p,_ in mapping])
manifest=dict(source_git_commit=sha,bucket_prefix=prefix,classification=metadata['classification'],
    checkpoint_completed_epochs=ITEM['history_rows'],configured_epochs=ITEM['epochs'],new_evaluation_performed=False,
    files=[dict(source=str(p),remote=dest,bytes=h.file_size,xet_hash=h.hash) for (p,dest),h in zip(mapping,hashes)])
(stage/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\\n')
print(json.dumps(manifest))
''',timeout=180)
    save('historical_backup/'+prefix+'_manifest.json',manifest)
    stage='/root/autodl-tmp/text_decoder_runs/historical_backup/'+prefix
    command='env PYTHONPATH=/root/autodl-tmp/hf-bucket-client '+PYTHON+' '+stage+'/upload.py --manifest '+stage+'/manifest.json'
    print(json.dumps(dict(event='historical_checkpoint_backup',prefix=prefix,bytes=sum(f['bytes'] for f in manifest['files']))),flush=True)
    p=subprocess.run(['ssh','-i',KEY,'-p','21465','-o','BatchMode=yes',HOST,command],
        input=token+'\n',text=True,encoding='utf-8',capture_output=True,timeout=1200)
    log=re.sub(r'hf_[A-Za-z0-9]{16,}','[REDACTED]',(p.stdout+'\n'+p.stderr).replace(token,'[REDACTED]'))
    (HERE/'historical_backup'/ (prefix+'_upload.log')).write_text(log,encoding='utf-8')
    assert p.returncode==0,'Backup failed; see sanitized log'
    proof=json.loads(p.stdout.splitlines()[-1]);assert proof['verified']
    actual={f.path:f for f in api.list_bucket_tree('razaxq/BetterLViT',prefix=prefix,recursive=True) if getattr(f,'xet_hash',None)}
    expected={f['remote']:f for f in manifest['files']};assert set(actual)==set(expected)
    for path,x in expected.items():assert actual[path].size==x['bytes'] and actual[path].xet_hash==x['xet_hash']
    proof.update(independent_local_listing_verified=True,completed_unix=time.time())
    save('historical_backup/'+prefix+'_verified.json',proof)
    print(json.dumps(dict(event='historical_backup_verified',prefix=prefix,files=len(expected))),flush=True)

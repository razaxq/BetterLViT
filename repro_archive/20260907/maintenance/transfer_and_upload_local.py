"""Resumable local relay with source/local/HF Xet and size verification."""
import json
import os
from pathlib import Path
import subprocess
import sys
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
os.environ['HF_XET_CACHE']='D:/BetterLViT/outputs/maintenance_20260907/local_xet_cache'
import hf_xet
from huggingface_hub import HfApi,get_token

root=Path(__file__).resolve().parent
manifest=json.loads((root/'transfer_manifest.json').read_text())
api=HfApi(token=get_token())
bucket='razaxq/BetterLViT'
local_by_hash={}
verified=[]
for arm in manifest:
    prefix=arm['source_git_commit'][:8]
    remote={x.path:x for x in api.list_bucket_tree(bucket,prefix=prefix,recursive=True) if getattr(x,'xet_hash',None)}
    expected={x['remote'] for x in arm['files']}
    assert set(remote)<=expected
    for item in arm['files']:
        target=root/'local_model_backup'/item['remote']
        target.parent.mkdir(parents=True,exist_ok=True)
        previous=local_by_hash.get(item['xet_hash'])
        if not target.exists() and previous:
            os.link(previous,target)
        if not target.exists() or target.stat().st_size!=item['bytes']:
            source=item['source']
            assert source.startswith('/root/') and not any(c in source for c in (' ',"'",'"',';','\n'))
            print('DOWNLOAD_BEGIN',item['remote'],item['bytes'],flush=True)
            subprocess.run(['scp','-i','C:/Users/dtftn/.ssh/seetacloud_betterlvit_ed25519_4090d',
                            '-P','21465','-o','BatchMode=yes','-o','ServerAliveInterval=15',
                            'root@connect.westb.seetacloud.com:'+source,str(target)],check=True)
        info=hf_xet.hash_files([str(target)])[0]
        assert info.hash==item['xet_hash'] and target.stat().st_size==item['bytes']==info.file_size
        local_by_hash[item['xet_hash']]=target
        print('LOCAL_VERIFIED',item['remote'],flush=True)
        existing=remote.get(item['remote'])
        if existing:
            assert existing.size==item['bytes'] and existing.xet_hash==item['xet_hash']
        else:
            same=next((x for x in remote.values() if x.xet_hash==item['xet_hash']),None)
            if same:
                api.batch_bucket_files(bucket,copy=[('bucket',bucket,item['xet_hash'],item['remote'])])
            else:
                print('UPLOAD_BEGIN',item['remote'],flush=True)
                api.batch_bucket_files(bucket,add=[(str(target),item['remote'])])
        remote={x.path:x for x in api.list_bucket_tree(bucket,prefix=prefix,recursive=True) if getattr(x,'xet_hash',None)}
        got=remote[item['remote']]
        assert got.xet_hash==item['xet_hash'] and got.size==item['bytes']
        print('HF_VERIFIED',item['remote'],flush=True)
    assert set(remote)==expected
    arm['verified']=True
    arm['classification']='completed_exploratory_80e'
    verified.append(arm)
    (root/'new_runs_upload_verified.json').write_text(json.dumps(verified,indent=2),encoding='utf-8')
    print('ARM_COMPLETE',arm['arm'],len(arm['files']),flush=True)
print('ALL_VERIFIED',len(verified),sum(len(x['files']) for x in verified),flush=True)

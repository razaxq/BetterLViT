"""Audit and remove only cloud-verified inactive completed Last checkpoint copies."""
import argparse
import json
import time
from pathlib import Path
from huggingface_hub import HfApi,get_token
from remote_ops import HERE,DOCS,remote,read,save
def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('audit','apply'));action=ap.parse_args().action
    if action=='audit':
        items=[];api=HfApi(token=get_token())
        for directory,label in [('regional_supervision_execution','rs2'),('regional_supervision_execution','rs3'),
                                ('visual_aux_execution','s1'),('visual_aux_execution','s2')]:
            folder=DOCS/'repro_archive/20260911'/directory/(label+'_results')
            if not folder.exists():folder=DOCS/'repro_archive/20260910'/directory/(label+'_results')
            m=read(folder/'hf_manifest.json');proof=read(folder/'hf_upload_verified.json')
            assert proof['verified'] and proof['independent_local_listing_verified']
            candidates={role:next(f for f in m['files'] if f['remote']==m['bucket_prefix']+'/models/'+role+'_model-BetterLViT.pth.tar') for role in ('best','last')}
            actual={f.path:f for f in api.list_bucket_tree('razaxq/BetterLViT',prefix=m['bucket_prefix']+'/models',recursive=True) if getattr(f,'xet_hash',None)}
            for x in candidates.values():assert actual[x['remote']].size==x['bytes'] and actual[x['remote']].xet_hash==x['xet_hash']
            items.append(dict(label=label,source_git_commit=m['source_git_commit'],cloud_checked_unix=time.time(),**candidates))
        save('storage_cloud_verified.json',dict(verified=True,bucket='razaxq/BetterLViT',items=items))
    else:
        assert read(HERE/'storage_audit.json')['verified']
    items=read(HERE/'storage_cloud_verified.json')['items']
    result=remote('ITEMS='+repr(items)+'\nAPPLY='+repr(action=='apply')+'\n'+'''
import fcntl,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
import hf_xet,torch
allowed=Path('/root/autodl-tmp').resolve()
lock=Path('/root/autodl-tmp/regional_runs/regional_training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
assert subprocess.run(['pgrep','-f','[t]rain_model.py'],capture_output=True).returncode==1
def size():return dict(scratch_free_bytes=shutil.disk_usage(allowed).free,root_free_bytes=shutil.disk_usage('/root').free,
    fs_bytes=int(subprocess.check_output(['du','-sb','/autodl-fs/data'],text=True).split()[0]))
def refs(paths):
    values=set(str(p) for p in paths);found=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            for fd in (proc/'fd').iterdir():
                try:
                    if os.readlink(fd) in values:found.append(dict(pid=proc.name,fd=fd.name))
                except OSError:pass
            maps=(proc/'maps').read_text(errors='replace')
            if any(v in maps for v in values):found.append(dict(pid=proc.name,mapped=True))
        except (OSError,PermissionError):pass
    return found
before=size();assert before['fs_bytes']<20_000_000_000
verified=[]
for item in ITEMS:
    src=Path(item['last']['source']);best=Path(item['best']['source'])
    assert src.name=='last_model-BetterLViT.pth.tar' and best.name=='best_model-BetterLViT.pth.tar'
    assert src.parent==best.parent and src.is_file() and best.is_file() and not src.is_symlink() and not best.is_symlink()
    assert src.resolve().is_relative_to(allowed) and best.resolve().is_relative_to(allowed)
    assert src.stat().st_dev==allowed.stat().st_dev
    assert not refs([src,best])
    hashes=hf_xet.hash_files([str(src),str(best)])
    for role,h in zip(('last','best'),hashes):assert h.hash==item[role]['xet_hash'] and h.file_size==item[role]['bytes']
    sha256=hashlib.file_digest(src.open('rb'),'sha256').hexdigest()
    ck=torch.load(src,map_location='cpu',weights_only=True)
    assert ck['source_git_commit']==item['source_git_commit'] and ck['epoch']==79 and ck['epochs']==80 and len(ck['epoch_history'])==80
    del ck
    ck=torch.load(best,map_location='cpu',weights_only=True);assert ck['source_git_commit']==item['source_git_commit'];del ck
    verified.append(dict(label=item['label'],source_git_commit=item['source_git_commit'],path=str(src),best_path=str(best),
        bytes=src.stat().st_size,inode=src.stat().st_ino,mtime_ns=src.stat().st_mtime_ns,sha256=sha256,
        xet_hash=item['last']['xet_hash'],cloud_path=item['last']['remote'],best_xet_hash=item['best']['xet_hash']))
if APPLY:
    assert not refs([Path(x['path']) for x in verified])
    for x in verified:
        p=Path(x['path']);st=p.stat()
        assert (st.st_ino,st.st_size,st.st_mtime_ns)==(x['inode'],x['bytes'],x['mtime_ns'])
        p.unlink()
        assert not p.exists() and Path(x['best_path']).is_file()
after=size();assert after['fs_bytes']==before['fs_bytes']
if APPLY:assert after['scratch_free_bytes']>4_000_000_000
print(json.dumps(dict(verified=True,applied=APPLY,files=verified,logical_bytes=sum(x['bytes'] for x in verified),
    before=before,after=after,active_references=[],best_checkpoints_preserved=True,cache_and_data_preserved=True,completed_unix=time.time())))
''',timeout=240)
    save('storage_cleanup.json' if action=='apply' else 'storage_audit.json',result)
    print(json.dumps(result))
if __name__=='__main__':main()

"""Delete five completed, cloud-verified historical Last copies; preserve Best."""
import argparse
import json
import time
from huggingface_hub import HfApi, get_token
from remote_ops import HERE, read, remote, save
ap=argparse.ArgumentParser();ap.add_argument('action',choices=('audit','apply'));action=ap.parse_args().action
if action=='audit':
    items=read(HERE/'storage_candidates.json')[:5]
    hashes=remote('ITEMS='+repr(items)+'\n'+'''
import json,hf_xet
out=[]
for item in ITEMS:
    for role in ('best','last'):
        h=hf_xet.hash_files([item[role]])[0]
        out.append(dict(label=item['label'],role=role,path=item[role],bytes=h.file_size,xet_hash=h.hash))
print(json.dumps(out))
''')
    api=HfApi(token=get_token())
    # Earlier pilots were preserved in the whole-project backup before the
    # short-SHA layout was adopted. Match their exact Xet hash, never just name.
    backup_files=None
    for item in items:
        assert item['epoch']+1==item['epochs']==item['history_rows']
        prefix=item['source_git_commit'][:8]
        files=list(api.list_bucket_tree('razaxq/BetterLViT',prefix=prefix,recursive=True))
        item['verified_files']={}
        for role in ('best','last'):
            local=next(x for x in hashes if x['label']==item['label'] and x['role']==role)
            matches=[f for f in files if f.path.endswith('/'+role+'_model-BetterLViT.pth.tar')
                     and getattr(f,'xet_hash',None)==local['xet_hash'] and f.size==local['bytes']]
            if not matches:
                if backup_files is None:
                    backup_files=list(api.list_bucket_tree('razaxq/BetterLViT',prefix='backup',recursive=True))
                matches=[f for f in backup_files if f.path.endswith('/'+role+'_model-BetterLViT.pth.tar')
                         and getattr(f,'xet_hash',None)==local['xet_hash'] and f.size==local['bytes']]
            assert len(matches)==1,(item['label'],role,'Fresh cloud hash match missing')
            item['verified_files'][role]=dict(**local,remote=matches[0].path)
    save('storage_cloud_verified.json',dict(verified=True,checked_unix=time.time(),items=items))
else:
    assert not (HERE/'storage_cleanup.json').exists(),'Cleanup already recorded'
    proof=read(HERE/'storage_cloud_verified.json');assert proof['verified']
    assert time.time()-proof['checked_unix']<3600
    items=proof['items']
result=remote('ITEMS='+repr(items)+'\nAPPLY='+repr(action=='apply')+'\n'+'''
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
import hf_xet,torch
allowed=Path('/root/autodl-tmp').resolve()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert subprocess.run(['pgrep','-f','[t]rain_model.py'],capture_output=True).returncode==1
def refs(paths):
    paths=set(map(str,paths));found=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            for fd in (proc/'fd').iterdir():
                try:
                    if os.readlink(fd) in paths:found.append(str(fd))
                except OSError:pass
            maps=(proc/'maps').read_text(errors='replace')
            if any(p in maps for p in paths):found.append(str(proc/'maps'))
        except (OSError,PermissionError):pass
    return found
def usage():
    return dict(scratch_free_bytes=shutil.disk_usage(allowed).free,
        shared_bytes=int(subprocess.check_output(['du','-sb','/autodl-fs/data'],text=True).split()[0]))
before=usage();assert before['shared_bytes']<20_000_000_000
verified=[]
for item in ITEMS:
    p=Path(item['last']);b=Path(item['best'])
    assert p.name=='last_model-BetterLViT.pth.tar' and b.name=='best_model-BetterLViT.pth.tar' and p.parent==b.parent
    assert not p.is_symlink() and not b.is_symlink()
    assert p.resolve().is_relative_to(allowed) and b.resolve().is_relative_to(allowed)
    assert p.stat().st_dev==allowed.stat().st_dev and not refs([p,b])
    for role in ('best','last'):
        h=hf_xet.hash_files([item[role]])[0];expected=item['verified_files'][role]
        assert h.file_size==expected['bytes'] and h.hash==expected['xet_hash']
        ck=torch.load(item[role],map_location='cpu',weights_only=True)
        assert ck['source_git_commit']==item['source_git_commit']
        if role=='last':assert ck['epoch']+1==ck['epochs']==len(ck['epoch_history'])
        del ck
    st=p.stat()
    verified.append(dict(path=str(p),best_path=str(b),bytes=st.st_size,inode=st.st_ino,mtime_ns=st.st_mtime_ns,
        sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),source_git_commit=item['source_git_commit']))
if APPLY:
    assert not refs([Path(x['path']) for x in verified])
    for x in verified:
        p=Path(x['path']);st=p.stat()
        assert (st.st_size,st.st_ino,st.st_mtime_ns)==(x['bytes'],x['inode'],x['mtime_ns'])
        p.unlink()
        assert not p.exists() and Path(x['best_path']).is_file()
after=usage();assert before['shared_bytes']==after['shared_bytes']
if APPLY:assert after['scratch_free_bytes']>6_000_000_000
print(json.dumps(dict(verified=True,applied=APPLY,before=before,after=after,files=verified,
    logical_bytes=sum(x['bytes'] for x in verified),best_preserved=True,rs1_models_and_caches_preserved=True,
    active_references=[],completed_unix=time.time())))
''',timeout=240)
save('storage_cleanup.json' if action=='apply' else 'storage_audit.json',result)
print(json.dumps(result))

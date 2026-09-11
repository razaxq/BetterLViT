"""Move one verified completed RS1 Last to root disk, preserving its old path.

Fresh HF path/size/Xet verification precedes copy. Check destination bytes and
checkpoint provenance before atomically replacing the source with a symlink.
No model is discarded; no shared-fs writes or recursive deletes are performed.
"""
import json
import time
from huggingface_hub import HfApi, get_token
from remote_ops import HERE, remote, save


def main():
    manifest=json.loads((HERE/'rs1_results/hf_manifest.json').read_text())
    proof=json.loads((HERE/'rs1_results/hf_upload_verified.json').read_text())
    assert proof['verified'] and proof['independent_local_listing_verified']
    assert proof['source_git_commit']==manifest['source_git_commit']
    assert json.loads((HERE/'rs1_final_snapshot.json').read_text())['runtime']['phase']=='complete'
    item=next(f for f in manifest['files'] if f['remote']==manifest['bucket_prefix']+'/models/last_model-BetterLViT.pth.tar')
    api=HfApi(token=get_token())
    actual=next(f for f in api.list_bucket_tree('razaxq/BetterLViT',prefix=manifest['bucket_prefix']+'/models',recursive=True) if f.path==item['remote'])
    assert actual.size==item['bytes'] and actual.xet_hash==item['xet_hash']
    cloud_checked=time.time()
    result=remote('ITEM='+repr(item)+'\nSHA='+repr(manifest['source_git_commit'])+'\n'+'''
import fcntl,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
import hf_xet
import torch
src=Path(ITEM['source'])
allowed=Path('/root/autodl-tmp/BetterLViT-regional-rs1').resolve()
assert src.is_absolute() and src.name=='last_model-BetterLViT.pth.tar'
assert src.parent.resolve().is_relative_to(allowed)
archive=Path('/root/regional_model_archive')
archive.mkdir(exist_ok=True)
assert archive.resolve()==archive and archive.stat().st_dev!=allowed.stat().st_dev
destination=archive/SHA[:8]/src.name
destination.parent.mkdir(exist_ok=True)
assert destination.parent.resolve().is_relative_to(archive)
receipt=destination.parent/'migration_verified.json'
lock=Path('/root/autodl-tmp/regional_runs/regional_training.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert subprocess.run(['pgrep','-f','[t]rain_model.py'],capture_output=True).returncode==1
def xet(path):
    info=hf_xet.hash_files([str(path)])[0]
    assert info.hash==ITEM['xet_hash'] and info.file_size==ITEM['bytes']
    return info.hash
def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def references():
    found=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            for fd in (proc/'fd').iterdir():
                try:
                    if os.readlink(fd)==str(src):found.append(dict(pid=int(proc.name),fd=fd.name))
                except OSError:pass
            if str(src) in (proc/'maps').read_text(errors='replace'):found.append(dict(pid=int(proc.name),mapped=True))
        except (PermissionError,FileNotFoundError,ProcessLookupError):pass
    return found
if src.is_symlink():
    assert src.resolve()==destination and receipt.exists()
    xet(src)
    result=json.loads(receipt.read_text());assert result['verified']
    result['replayed']=True
else:
    assert src.is_file() and src.resolve()==src
    refs=references();assert not refs,refs
    before=dict(scratch_free_bytes=shutil.disk_usage(allowed).free,root_free_bytes=shutil.disk_usage(archive).free,
        fs_bytes=int(subprocess.check_output(['du','-sb','/autodl-fs/data'],text=True).split()[0]))
    assert before['root_free_bytes']>ITEM['bytes']+2_000_000_000 and before['fs_bytes']<20_000_000_000
    signature=(src.stat().st_ino,src.stat().st_size,src.stat().st_mtime_ns)
    source_xet=xet(src);source_sha=sha256(src)
    if not destination.exists():
        partial=destination.with_name(destination.name+'.partial')
        assert not partial.exists()
        with src.open('rb') as incoming,partial.open('xb') as outgoing:
            shutil.copyfileobj(incoming,outgoing,length=8*1024*1024)
            outgoing.flush();os.fsync(outgoing.fileno())
        assert sha256(partial)==source_sha
        os.replace(partial,destination)
    assert destination.is_file() and not destination.is_symlink()
    assert xet(destination)==source_xet and sha256(destination)==source_sha
    checkpoint=torch.load(destination,map_location='cpu',weights_only=True)
    assert checkpoint['source_git_commit']==SHA and checkpoint['epoch']==79 and checkpoint['epochs']==80
    assert len(checkpoint['epoch_history'])==80
    checkpoint_metadata=dict(source_git_commit=checkpoint['source_git_commit'],epoch=checkpoint['epoch'],best_epoch=checkpoint['best_epoch'])
    del checkpoint
    assert signature==(src.stat().st_ino,src.stat().st_size,src.stat().st_mtime_ns)
    assert not references()
    link=src.with_name(src.name+'.verified-link')
    assert not link.exists() and not link.is_symlink()
    os.symlink(str(destination),str(link))
    os.replace(link,src)
    assert src.is_symlink() and src.resolve()==destination and xet(src)==source_xet
    sibling_best=src.with_name('best_model-BetterLViT.pth.tar')
    assert sibling_best.is_file() and not sibling_best.is_symlink()
    after=dict(scratch_free_bytes=shutil.disk_usage(allowed).free,root_free_bytes=shutil.disk_usage(archive).free,
        fs_bytes=int(subprocess.check_output(['du','-sb','/autodl-fs/data'],text=True).split()[0]))
    assert after['fs_bytes']==before['fs_bytes'] and after['scratch_free_bytes']>4_000_000_000
    result=dict(verified=True,source_git_commit=SHA,source_path=str(src),destination_path=str(destination),
        bytes=ITEM['bytes'],xet_hash=source_xet,sha256=source_sha,checkpoint=checkpoint_metadata,
        source_path_preserved=True,source_is_symlink=True,sibling_best_preserved=True,
        active_references_before=refs,before=before,after=after,completed_unix=time.time(),replayed=False)
    receipt.write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps(result))
''',timeout=240)
    result.update(fresh_cloud_listing_verified=True,cloud_checked_unix=cloud_checked,bucket='razaxq/BetterLViT',bucket_path=item['remote'])
    save('storage_migration_rs1_last.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

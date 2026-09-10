"""Short SSH operations over standard input, with structured local evidence."""
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
KEY='C:/Users/dtftn/.ssh/seetacloud_betterlvit_ed25519'
PYTHON='/root/autodl-tmp/envs/betterlvit-paper/bin/python'


def remote(code, timeout=120):
    result=subprocess.run(['ssh','-i',KEY,'-p','21465','-o','BatchMode=yes',
        '-o','ConnectTimeout=15','root@connect.westb.seetacloud.com',PYTHON+' -'],
        input=code,text=True,capture_output=True,timeout=timeout)
    if result.returncode: raise RuntimeError(result.stderr+result.stdout[-6000:])
    return json.loads(result.stdout)


def save(name,value):
    (HERE/name).parent.mkdir(parents=True,exist_ok=True)
    (HERE/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def storage_audit(remove=False):
    code='REMOVE='+repr(remove)+'\n'+'''
import json, os, shutil, subprocess, time
from pathlib import Path
import numpy as np
target=Path('/root/autodl-tmp/visual_probe_cache_20260908')
assert target.resolve()==target and target.is_dir() and not target.is_symlink()
expected={f'{kind}_{split}{suffix}.npy' for kind in ('cxformer','dinov2')
          for split in ('Train_Folder','Val_Folder') for suffix in ('','_masks')}
assert {p.name for p in target.iterdir()}==expected
files=[]
for path in sorted(target.iterdir()):
    assert path.is_file() and not path.is_symlink() and path.resolve().parent==target
    a=np.load(path,mmap_mode='r')
    files.append(dict(path=str(path),bytes=path.stat().st_size,shape=list(a.shape),dtype=str(a.dtype)))
    del a
opened=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name)==os.getpid(): continue
    try:
        for fd in (proc/'fd').iterdir():
            try:
                value=os.readlink(fd)
                if value.startswith(str(target)+'/'): opened.append(dict(pid=proc.name,fd=fd.name,target=value))
            except (OSError,FileNotFoundError): pass
        if str(target) in (proc/'maps').read_text(): opened.append(dict(pid=proc.name,mapped=True))
    except (OSError,FileNotFoundError): pass
assert not opened,opened
active=subprocess.run(['pgrep','-f','[t]rain_model.py|[p]robe_visual_prior.py'],capture_output=True,text=True)
assert active.returncode==1,active.stdout
before={p:shutil.disk_usage(p).free for p in ('/root','/root/autodl-tmp')}
fs_before=int(subprocess.check_output(['du','-sb','/autodl-fs/data'],text=True).split()[0])
assert fs_before<20_000_000_000
if REMOVE:
    for item in files: Path(item['path']).unlink()
    target.rmdir()
after={p:shutil.disk_usage(p).free for p in before}
fs_after=int(subprocess.check_output(['du','-sb','/autodl-fs/data'],text=True).split()[0])
assert fs_after<20_000_000_000
print(json.dumps(dict(time_unix=time.time(),target=str(target),files=files,
    logical_bytes=sum(p['bytes'] for p in files),active_users=opened,
    no_training_or_visual_probe_active=True,removed=REMOVE,free_before=before,free_after=after,
    fs_bytes_before=fs_before,fs_bytes_after=fs_after,
    regeneration_source='/root/BetterLViT-visual-prior-dev/tools/probe_visual_prior.py',
    reason='Derived feature and resized-mask cache only; models, original dataset and result records retained')))
'''
    value=remote(code)
    save('storage_cleanup.json' if remove else 'storage_audit.json',value)
    print(json.dumps(value,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--remove-derived-cache',action='store_true')
    storage_audit(parser.parse_args().remove_derived_cache)

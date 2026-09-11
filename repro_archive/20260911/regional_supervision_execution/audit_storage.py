"""Read-only disk inventory before authorized regional-run storage maintenance."""
import json
from remote_ops import remote, save


result = remote('''
import json,os,shutil,subprocess,time
from pathlib import Path
roots=['/root','/root/autodl-tmp','/autodl-fs/data']
usage={p:dict(zip(('total','used','free'),shutil.disk_usage(p))) for p in roots}
sizes={p:subprocess.check_output(['du','-x','-B1','--max-depth=1',p],text=True,stderr=subprocess.DEVNULL).splitlines() for p in roots[:2]}
large=[]
for folder,dirs,files in os.walk('/root/autodl-tmp',followlinks=False):
    for name in files:
        p=Path(folder)/name
        if p.is_symlink():continue
        try:
            st=p.stat()
            if st.st_size>=200_000_000:large.append(dict(path=str(p),bytes=st.st_size,device=st.st_dev))
        except FileNotFoundError:pass
open_files=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:
        command=(proc/'comm').read_text().strip()
        for fd in (proc/'fd').iterdir():
            try:
                target=os.readlink(fd)
                if target.startswith('/root/autodl-tmp/') or target.startswith('/root/recipe_runs/'):
                    open_files.append(dict(pid=int(proc.name),command=command,fd=fd.name,path=target))
            except (OSError,PermissionError):pass
    except (OSError,PermissionError):pass
fs=int(subprocess.check_output(['du','-sb','/autodl-fs/data'],text=True).split()[0])
print(json.dumps(dict(audited_unix=time.time(),usage=usage,depth_one_disk_bytes=sizes,
    large_files=sorted(large,key=lambda r:r['bytes'],reverse=True),open_files=open_files,fs_bytes=fs)))
''',timeout=180)
save('storage_before_rs3.json',result)
print(json.dumps(result,indent=2))

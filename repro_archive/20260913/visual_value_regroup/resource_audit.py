"""Read-only inventory of checkpoint sizes, free space and active GPU jobs."""
from remote_ops import remote,save
import json
result=remote('''
import json,os,shutil,subprocess,time
from pathlib import Path
root=Path('/root/autodl-tmp'); files=[]
for folder,dirs,names in os.walk(root):
    dirs[:]=[d for d in dirs if d not in ('envs','datasets','.git','huggingface')]
    for name in names:
        p=Path(folder)/name
        if p.is_file() and p.stat().st_size>100_000_000:
            files.append(dict(path=str(p),bytes=p.stat().st_size,symlink=p.is_symlink()))
print(json.dumps(dict(free=shutil.disk_usage(root).free,root_free=shutil.disk_usage('/root').free,
    shared_bytes=int(subprocess.check_output(['du','-sb','/autodl-fs/data'],text=True).split()[0]),
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True),
    files=files,checked_unix=time.time())))
''')
save('resource_audit.json',result)
print(json.dumps(result))

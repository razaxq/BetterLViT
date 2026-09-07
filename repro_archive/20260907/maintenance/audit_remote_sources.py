import json
from pathlib import Path
import subprocess

base='/root/autodl-tmp/BetterLViT-paper-ablation'
listing=subprocess.check_output(['git','-C',base,'worktree','list','--porcelain'],text=True)
result=[]
for line in listing.splitlines():
    if not line.startswith('worktree '): continue
    path=line[len('worktree '):]
    status=subprocess.check_output(['git','-C',path,'status','--short'],text=True)
    commit=subprocess.check_output(['git','-C',path,'rev-parse','HEAD'],text=True).strip()
    result.append({'path':path,'commit':commit,'status':status})
Path('/root/maintenance_20260907/remote_source_audit.json').write_text(json.dumps(result,indent=2))
print('WORKTREES',len(result))
print(json.dumps([x for x in result if x['status']],indent=2))

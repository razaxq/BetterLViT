"""Delete only unchanged, verified HF-backed Last files; retain all Best files."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

if sys.argv[1:] != ['--execute']:
    raise SystemExit('Requires --execute after reviewing cleanup_plan.json')
out=Path('/root/maintenance_20260907')
plan=json.loads((out/'cleanup_plan.json').read_text())
root=Path('/root/autodl-tmp').resolve()
targets={Path(x['path']).resolve():x for x in plan['verified_candidates']}
assert targets
for p,row in targets.items():
    assert p.is_relative_to(root) and p.name=='last_model-BetterLViT.pth.tar'
    assert p.parent.name=='models' and Path(row['best_preserved']).is_file()
    stat=p.stat()
    assert (stat.st_size,stat.st_mtime_ns,stat.st_ino)==(row['bytes'],row['mtime_ns'],row['inode'])
active=[]; opened=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name)==os.getpid(): continue
    try:
        args=(proc/'cmdline').read_bytes().split(b'\0')
        if any(Path(a.decode(errors='ignore')).name in ('train_model.py','evaluate_experiment.py') for a in args):
            active.append(proc.name)
        for fd in (proc/'fd').iterdir():
            try:
                if fd.resolve() in targets: opened.append((proc.name,str(fd.resolve())))
            except (OSError,RuntimeError): pass
    except (OSError,PermissionError): pass
assert not active, ('training_or_evaluation_active',active)
assert not opened, ('target_open_files',opened)
def disk(): return dict(zip(('total','used','free'),shutil.disk_usage(root)))
def shared_bytes():
    return int(subprocess.check_output(['du','-s','--apparent-size','-B1','/root/autodl-fs/'],text=True).split()[0])
report={'before':disk(),'shared_apparent_bytes_before':shared_bytes(),'deleted':[],
        'policy':'HF Xet + size verified; unchanged inode/mtime; no active train/eval; no target open handles; all Best retained'}
assert report['shared_apparent_bytes_before']<20_000_000_000
report_path=out/'cleanup_completed.json'
report_path.write_text(json.dumps(report,indent=2))
for p,row in targets.items():
    p.unlink()
    assert Path(row['best_preserved']).is_file()
    report['deleted'].append(row)
    report_path.write_text(json.dumps(report,indent=2))
report['after']=disk()
report['shared_apparent_bytes_after']=shared_bytes()
report['logical_bytes_removed']=sum(x['bytes'] for x in report['deleted'])
report['free_bytes_increase']=report['after']['free']-report['before']['free']
report['complete']=True
report_path.write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='deleted'},indent=2))

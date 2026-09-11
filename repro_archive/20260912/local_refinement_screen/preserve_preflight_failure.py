"""Archive failed source/receipts; remove only its unusable generated point cache."""
import base64
import json
from pathlib import Path
from control import remote
from analysis import write_json
HERE=Path(__file__).resolve().parent
d=json.loads((HERE/'deployment.json').read_text())
assert d['source_git_commit']=='7c9ec09f664d4faac068da3d2cb2ad46a46c352f'
assert json.loads((HERE/'preflight_collection.json').read_text())['runtime']['phase']=='failed'
result=remote('ROOT='+repr(d['remote_directory'])+'\nPID='+repr(d['pid'])+'\n'+'''
import base64,json,shutil
from pathlib import Path
root=Path(ROOT).resolve();assert root==Path('/root/local_refinement_f_7c9ec09f')
runtime=json.loads((root/'preflight/runtime.json').read_text());assert runtime['phase']=='failed'
# Audit every active process and FD before removing this attempt's generated cache.
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:
        command=(proc/'cmdline').read_bytes()
        assert not (str(root/'preflight.py').encode() in command),'Failed preflight still active'
        for fd in (proc/'fd').iterdir():
            try:target=fd.resolve()
            except OSError:continue
            assert not target.is_relative_to(root/'cache'),'A process still holds this cache'
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
files={}
for name in ('synthetic_checks.json','runtime.json'):
    files[name]=base64.b64encode((root/'preflight'/name).read_bytes()).decode()
before=shutil.disk_usage(root).free;removed=[]
for name in ('fine.bin','indices.bin'):
    target=(root/'cache'/name).resolve();assert target.parent==root/'cache' and target.is_file()
    st=target.stat();removed.append(dict(path=str(target),logical_bytes=st.st_size,allocated_bytes=st.st_blocks*512));target.unlink()
assert not list((root/'cache').iterdir());(root/'cache').rmdir()
print(json.dumps(dict(files=files,cleanup=dict(removed=removed,free_before_bytes=before,free_after_bytes=shutil.disk_usage(root).free,
    active_process_and_open_file_audit=True,only_failed_attempt_generated_cache=True,old_B_cache_untouched=True))))
''')
target=HERE/'attempt_v1';target.mkdir(exist_ok=False)
files=result.pop('files')
for name,value in files.items():(target/name).write_bytes(base64.b64decode(value))
for name in ('deployment.json','deployment_attempt.json','preflight_collection.json'):(HERE/name).replace(target/name)
write_json(target/'repair.json',dict(**result,cause='Native labels are [B,H,W], packed cache expands to [B,1,H,W]; explicit unsqueeze preserves labels',
    source_git_commit=d['source_git_commit'],formal_training_started=False,metric_tolerance_unchanged=True))
print(json.dumps(result))

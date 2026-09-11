"""Dispatch one bounded CPU artifact diagnosis, without inspecting the completed run."""
import base64
import json
from pathlib import Path
import subprocess
from control import remote,DOCS,HERE
from analysis import digest,write_json

sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=DOCS,text=True).strip()
path=HERE/'diagnose_collapse.py'
blob=subprocess.check_output(['git','show',sha+':'+path.relative_to(DOCS).as_posix()],cwd=DOCS)
assert blob==path.read_bytes()
code='SCRIPT='+repr(base64.b64encode(blob).decode())+'\n' + '''
import base64,json,os,subprocess
from pathlib import Path
root=Path('/root/text_head_b_49905dbb');path=root/'diagnose_collapse.py'
data=base64.b64decode(SCRIPT)
if path.exists():assert path.read_bytes()==data
else:path.write_bytes(data)
r=subprocess.run(['/root/autodl-tmp/envs/betterlvit-paper/bin/python','-B',str(path)],
    cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONHASHSEED='1219'),capture_output=True,text=True,timeout=60)
assert r.returncode==0,r.stderr+r.stdout
print(r.stdout)
'''
result=remote(code,timeout=90)
result.update(diagnostic_code_source_git_commit=sha,diagnostic_script_sha256=digest(path))
write_json(HERE/'posthoc_collapse.json',result)
print(json.dumps(dict(checkpoint_load_and_hash_verified=result['checkpoint_load_and_hash_verified'],
    model_updates=result['model_updates'],internal_holdout_evaluated=result['internal_holdout_evaluated'],
    parameters={v:{p:dict(initial=details['initial']['parameters'][p]['parameter_norm'],
        final=details['final']['parameters'][p]['parameter_norm'],
        initial_l2_task_ratio=details['initial']['parameters'][p]['l2_to_task_norm_ratio'])
        for p in details['initial']['parameters']} for v,details in result['variants'].items()}),indent=2))

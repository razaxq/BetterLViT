"""Deploy a committed CPU-only Train mask audit through one short SSH call."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
DOCS=HERE.parents[2]
sys.path.insert(0,str(HERE.parents[1]/'20260910/visual_aux_execution'))
from remote_ops import remote

if __name__=='__main__':
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=DOCS,text=True).strip()
    rel=(HERE/'audit_train_regions.py').relative_to(DOCS).as_posix()
    blob=subprocess.check_output(['git','show',sha+':'+rel],cwd=DOCS)
    expected=hashlib.sha256(blob).hexdigest()
    result=remote('DATA='+repr(base64.b64encode(blob).decode())+'\n'+'''
import base64,json,subprocess
from pathlib import Path
path=Path('/root/autodl-tmp/regional_supervision_audit_20260911.py')
assert not path.exists(), 'Do not overwrite a prior audit'
path.write_bytes(base64.b64decode(DATA))
p=subprocess.run(['/root/autodl-tmp/envs/betterlvit-paper/bin/python',str(path)],capture_output=True,text=True,timeout=120)
assert p.returncode==0,p.stderr
print(p.stdout)
''',timeout=150)
    assert result['script_sha256']==expected
    result['audit_source_git_commit']=sha
    (HERE/'train_region_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('examples','patches')},indent=2))

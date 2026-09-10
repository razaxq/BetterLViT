"""Run the bounded read-only Train audit without opening a training status connection."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent
EXE=HERE.parents[1]/'20260910/visual_aux_execution'
sys.path.insert(0,str(EXE))
from remote_ops import remote

script=HERE/'audit_targets.py';data=script.read_text(encoding='utf-8')
destination='/root/autodl-tmp/s2_target_audit_20260911.py'
result=remote('from pathlib import Path\nimport runpy\np=Path('+repr(destination)+')\np.write_text('+repr(data)+',encoding="utf-8")\nrunpy.run_path(str(p),run_name="__main__")\n',timeout=180)
assert result['script_sha256']==hashlib.sha256(data.encode()).hexdigest()
result['diagnostic_git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()
(HERE/'target_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='examples'}))

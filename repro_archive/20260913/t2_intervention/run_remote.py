"""Detached diagnostic execution with bounded collection, preserving frozen T2."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'text_decoder_controls'))
from remote_ops import remote,copy_to_remote,environment,PYTHON,HOST,KEY
RUN = '/root/autodl-tmp/t2_intervention_20260913'

def save(name,data):
    (HERE/name).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('launch','collect'));a=ap.parse_args()
    if a.mode == 'launch':
        source=json.loads((HERE.parent/'text_decoder_controls/sources.json').read_text())['t2']
        sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=DOCS,text=True).strip()
        assert not subprocess.check_output(['git','status','--porcelain','--',str(HERE.relative_to(DOCS))],cwd=DOCS,text=True).strip()
        audit=remote('''import json,subprocess,shutil
from pathlib import Path
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not gpu,'GPU in use'
p=Path('''+repr(RUN)+''');p.mkdir(exist_ok=False)
print(json.dumps({'gpu_idle':True,'disk':shutil.disk_usage(p)._asdict()}))''',timeout=40)
        for name in ('diagnose.py',):copy_to_remote(HERE/name,RUN+'/'+name)
        copy_to_remote(HERE.parent/'text_decoder_controls/t2_results/validation.json',RUN+'/reference.json')
        env=environment(source);env['ANALYSIS_GIT_COMMIT']=sha
        cmd=[PYTHON,RUN+'/diagnose.py','--repository',source['repository'],'--reference',RUN+'/reference.json','--output',RUN+'/result.json']
        receipt=remote('RUN='+repr(RUN)+'\nENV='+repr(env)+'\nCMD='+repr(cmd)+'\nCWD='+repr(source['repository'])+'\n'+'''
import os,json,subprocess,time,hashlib
from pathlib import Path
script=Path(RUN)/'diagnose.py'
log=open(Path(RUN)/'run.log','wb')
p=subprocess.Popen(CMD,cwd=CWD,env=dict(os.environ,**ENV),stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
r={'pid':p.pid,'started_unix':time.time(),'analysis_script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'command':CMD}
(Path(RUN)/'launch.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r))''',timeout=40)
        assert receipt['analysis_script_sha256']==hashlib.sha256((HERE/'diagnose.py').read_bytes()).hexdigest()
        receipt.update(analysis_git_commit=sha,resource_audit=audit,remote_run=RUN)
        save('launch.json',receipt);print(json.dumps(receipt))
    else:
        receipt=json.loads((HERE/'launch.json').read_text())
        value=remote('RUN='+repr(RUN)+'\nPID='+str(receipt['pid'])+'\n'+'''
import json,time
from pathlib import Path
p=Path(RUN);f=p/'result.json'
proc=Path('/proc')/str(PID)
print(json.dumps({'checked_unix':time.time(),'complete':f.exists(),'process_exists':proc.exists(),'log_tail':(p/'run.log').read_text(errors='replace')[-3500:]}))''',timeout=40)
        save('collection.json',value)
        if value['complete']:
            for name in ('result.json','run.log'):
                subprocess.run(['scp','-i',KEY,'-P','21465','-o','BatchMode=yes',HOST+':'+RUN+'/'+name,str(HERE/name)],check=True,capture_output=True)
        print(json.dumps(value))

if __name__=='__main__':main()

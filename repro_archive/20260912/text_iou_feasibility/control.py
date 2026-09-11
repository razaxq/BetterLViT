"""Immutable deployment and one predicted collection for a no-update audit."""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import time
from analysis import digest,write_json

HERE=Path(__file__).resolve().parent
DOCS=next(p for p in HERE.parents if (p/'.git').exists())
PYTHON='/root/autodl-tmp/envs/betterlvit-paper/bin/python'
FILES=('run_audit.py','heads.py','analysis.py','manifest.json','selection.json','PROTOCOL.md')

def remote(code,timeout=120):
    result=subprocess.run(['ssh','-i','C:/Users/dtftn/.ssh/seetacloud_betterlvit_ed25519','-p','21465',
        '-o','BatchMode=yes','-o','ConnectTimeout=15','root@connect.westb.seetacloud.com',PYTHON+' -'],
        input=code,text=True,encoding='utf-8',capture_output=True,timeout=timeout)
    if result.returncode:raise RuntimeError(result.stderr+'\n'+result.stdout[-8000:])
    return json.loads(result.stdout)

def launch():
    assert not (HERE/'launch_attempt.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=DOCS,text=True).strip()
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=DOCS,text=True).strip()
    payload={}
    for name in FILES:
        raw=subprocess.check_output(['git','show',sha+':'+(HERE/name).relative_to(DOCS).as_posix()],cwd=DOCS)
        assert raw==(HERE/name).read_bytes()
        payload[name]=base64.b64encode(raw).decode()
    root='/root/text_iou_e_'+sha[:8]
    write_json(HERE/'launch_attempt.json',dict(source_git_commit=sha,remote_directory=root,submitted_unix=time.time()))
    result=remote('ROOT='+repr(root)+'\nSHA='+repr(sha)+'\nPAYLOAD='+repr(payload)+'\n'+'''
import base64,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
root=Path(ROOT);root.mkdir(exist_ok=False)
assert shutil.disk_usage(root).free>128000000
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'GPU occupied before audit'
hashes={}
for name,value in PAYLOAD.items():
    assert Path(name).name==name
    raw=base64.b64decode(value);(root/name).write_bytes(raw);hashes[name]=hashlib.sha256(raw).hexdigest()
started=time.time()
with (root/'run.log').open('x') as log:
    p=subprocess.Popen(['/root/autodl-tmp/envs/betterlvit-paper/bin/python','-B','-u',str(root/'run_audit.py'),
        '--output',str(root/'results'),'--source-sha',SHA],cwd=root,
        env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='1219'),
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
receipt=dict(source_git_commit=SHA,remote_directory=ROOT,pid=p.pid,submitted_unix=started,
    files_sha256=hashes,train_updates=0,system_free_bytes=shutil.disk_usage(root).free,predicted_collect_unix=started+120)
(root/'launch.json').write_text(json.dumps(receipt));print(json.dumps(receipt))
''')
    assert result['files_sha256']=={n:digest(HERE/n) for n in FILES}
    write_json(HERE/'launch.json',result);print(json.dumps(result))

def collect():
    d=json.loads((HERE/'launch.json').read_text())
    assert time.time()>=d['predicted_collect_unix']
    assert not (HERE/'results').exists()
    result=remote('ROOT='+repr(d['remote_directory'])+'\n'+'''
import base64,hashlib,json,time
from pathlib import Path
root=Path(ROOT);runtime=json.loads((root/'results/runtime.json').read_text());out=dict(runtime=runtime,observed_unix=time.time(),files={})
if runtime['phase']=='complete':
    for name,meta in runtime['artifacts'].items():
        data=(root/'results'/name).read_bytes();assert hashlib.sha256(data).hexdigest()==meta['sha256']
        assert len(data)==meta['bytes'];out['files'][name]=base64.b64encode(data).decode()
out['log']=(root/'run.log').read_text(errors='replace');print(json.dumps(out))
''')
    raw=result.pop('files');write_json(HERE/'collection.json',result)
    if result['runtime']['phase']=='complete':
        (HERE/'results').mkdir()
        for name,value in raw.items():
            assert Path(name).name==name
            (HERE/'results'/name).write_bytes(base64.b64decode(value))
        write_json(HERE/'results/runtime.json',result['runtime'])
        (HERE/'results/run.log').write_text(result['log'],encoding='utf-8',newline='\n')
    print(json.dumps(dict(phase=result['runtime']['phase'],source_git_commit=d['source_git_commit'],
        elapsed_seconds=result['observed_unix']-d['submitted_unix'],log=result['log'][-4000:])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('launch','collect'));args=p.parse_args()
    globals()[args.action]()

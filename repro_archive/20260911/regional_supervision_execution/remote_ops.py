"""Bounded SSH calls and evidence local to the regional experiment series."""
import json
from pathlib import Path
import subprocess
HERE=Path(__file__).resolve().parent
KEY='C:/Users/dtftn/.ssh/seetacloud_betterlvit_ed25519'
HOST='root@connect.westb.seetacloud.com'
PYTHON='/root/autodl-tmp/envs/betterlvit-paper/bin/python'


def remote(code,timeout=180):
    p=subprocess.run(['ssh','-i',KEY,'-p','21465','-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,PYTHON+' -'],
        input=code,text=True,capture_output=True,timeout=timeout)
    if p.returncode:raise RuntimeError(p.stderr+p.stdout[-8000:])
    return json.loads(p.stdout)


def save(name,value):
    p=HERE/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def copy_to_remote(local,destination):
    subprocess.run(['scp','-i',KEY,'-P','21465','-o','BatchMode=yes',str(local),HOST+':'+destination],check=True,capture_output=True)


def environment(sha,profile):
    cache='/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface'
    return dict(BETTERLVIT_EXPERIMENT=profile,BETTERLVIT_GIT_COMMIT=sha,
        BETTERLVIT_SEED='1219',BETTERLVIT_EPOCHS='80',BETTERLVIT_BATCH_SIZE='16',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='1219',TOKENIZERS_PARALLELISM='false',
        HF_HOME=cache,HF_HUB_CACHE=cache+'/hub',HF_MODULES_CACHE=cache+'/modules',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TEST_SPLIT_ALLOWED='0',AUTO_TEST_EVALUATE='0')

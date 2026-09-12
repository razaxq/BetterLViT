"""Short SSH commands and evidence isolated from all historical runs."""
import json
from pathlib import Path
import subprocess
HERE=Path(__file__).resolve().parent
DOCS=HERE.parents[2]
KEY='C:/Users/dtftn/.ssh/seetacloud_betterlvit_ed25519'
HOST='root@connect.westb.seetacloud.com'
PYTHON='/root/autodl-tmp/envs/betterlvit-paper/bin/python'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(name,value):
    p=HERE/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
def remote(code,timeout=180):
    p=subprocess.run(['ssh','-i',KEY,'-p','21465','-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,PYTHON+' -'],
        input=code,text=True,encoding='utf-8',capture_output=True,timeout=timeout)
    if p.returncode:raise RuntimeError(p.stderr+p.stdout[-5000:])
    return json.loads(p.stdout)
def environment(source):
    cache='/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface'
    return dict(BETTERLVIT_EXPERIMENT='rs1_global_iou',BETTERLVIT_GIT_COMMIT=source['source_git_commit'],
        BETTERLVIT_SEED=str(source['seed']),BETTERLVIT_EPOCHS='80',BETTERLVIT_BATCH_SIZE='16',
        BETTERLVIT_CUDNN_ENABLED='1',BETTERLVIT_DETERMINISTIC='1',BETTERLVIT_RESUME_PATH='',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED=str(source['seed']),TOKENIZERS_PARALLELISM='false',
        HF_HOME=cache,HF_HUB_CACHE=cache+'/hub',HF_MODULES_CACHE=cache+'/modules',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TEST_SPLIT_ALLOWED='0',AUTO_TEST_EVALUATE='0')

def copy_to_remote(path,destination):
    subprocess.run(['scp','-i',KEY,'-P','21465','-o','BatchMode=yes',str(path),HOST+':'+destination],check=True,capture_output=True)

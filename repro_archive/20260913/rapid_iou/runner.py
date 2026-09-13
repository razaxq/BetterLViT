"""Detached evaluation runner, no training connection or automatic Test search."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
split=sys.argv[1];assert split in ('validation','test')
plan=json.loads((root/'plan.json').read_text())
deployment=json.loads((root/'deployment.json').read_text())
for name,digest in deployment['sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
source=plan['sources'][0]
cache='/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface'
env=os.environ.copy()
env.update(BETTERLVIT_EXPERIMENT='r2_single_cosine',BETTERLVIT_GIT_COMMIT=source['source_git_commit'],
    BETTERLVIT_SEED='1219',BETTERLVIT_EPOCHS='80',BETTERLVIT_BATCH_SIZE='16',BETTERLVIT_CUDNN_ENABLED='1',
    BETTERLVIT_DETERMINISTIC='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='1219',
    HF_HOME=cache,HF_HUB_CACHE=cache+'/hub',HF_MODULES_CACHE=cache+'/modules',HF_HUB_OFFLINE='1',
    TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',
    RAPID_EVALUATION_COMMIT=deployment['evaluation_source_git_commit'],RAPID_TEST_ALLOWED='1' if split=='test' else '0')
run=root/split
state=dict(phase='evaluation',split=split,evaluation_source_git_commit=deployment['evaluation_source_git_commit'],
    started_unix=time.time(),no_training=True)
def save():
    temp=run/'runtime.tmp';temp.write_text(json.dumps(state)+'\n');temp.replace(run/'runtime.json')
save()
cmd=[sys.executable,str(root/'evaluate.py'),'--split',split]
if split=='test':cmd+=['--selection',str(root/'selection.json')]
with (run/'evaluation.log').open('w') as log:
    rc=subprocess.run(cmd,env=env,cwd=source['repository'],stdout=log,stderr=subprocess.STDOUT).returncode
state.update(phase='complete' if rc==0 else 'failed',returncode=rc,ended_unix=time.time())
save()

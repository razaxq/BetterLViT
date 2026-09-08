"""Deploy once or run one bounded pre-training GPU check on a frozen source."""
import argparse
import json
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path('D:/BetterLViT/outputs/recipe_replication_20260909')
KEY='C:/Users/dtftn/.ssh/seetacloud_betterlvit_ed25519'
PYTHON='/root/autodl-tmp/envs/betterlvit-paper/bin/python'


def remote(code,timeout=60):
    value=subprocess.run(['ssh','-i',KEY,'-p','21465','-o','BatchMode=yes','-o','ConnectTimeout=15',
        'root@connect.westb.seetacloud.com',PYTHON+' -'],input=code,text=True,capture_output=True,timeout=timeout)
    if value.returncode:raise RuntimeError(value.stderr+value.stdout[-4000:])
    return json.loads(value.stdout)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('deploy','preflight'))
    p.add_argument('--label')
    p.add_argument('--repetition',type=int,choices=(1,2))
    args=p.parse_args()
    sources=json.loads((HERE/'sources.json').read_text())
    if args.action=='deploy':
        subprocess.run(['scp','-i',KEY,'-P','21465','-o','BatchMode=yes','D:/BetterLViT/outputs/recipe_replication.bundle',
            'root@connect.westb.seetacloud.com:/root/recipe_replication.bundle'],check=True)
        value=remote('SOURCES='+repr(sources)+'\n'+'''
import json,shutil,subprocess
from pathlib import Path
base=Path('/root/BetterLViT-visual-prior-dev')
assert subprocess.run(['pgrep','-f','[t]rain_model.py'],capture_output=True).returncode!=0
free=shutil.disk_usage('/root').free
assert free>9_500_000_000, 'Insufficient reserved space for all four new runs'
fs=int(subprocess.check_output(['du','-sb','/autodl-fs/data'],text=True).split()[0])
assert fs<20_000_000_000
for label,s in SOURCES.items():
    repo=Path(s['repository'])
    assert not repo.exists()
    subprocess.run(['git','fetch','/root/recipe_replication.bundle',s['branch']+':'+s['branch']],cwd=base,check=True,stdout=subprocess.PIPE)
    subprocess.run(['git','worktree','add',str(repo),s['branch']],cwd=base,check=True,stdout=subprocess.PIPE)
    (repo/'datasets').symlink_to('/root/autodl-tmp/datasets',target_is_directory=True)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==s['source_git_commit']
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
print(json.dumps(dict(sources=SOURCES,system_free_bytes_before=free,system_free_bytes_after=shutil.disk_usage('/root').free,
    shared_fs_bytes=fs,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total','--format=csv,noheader'],text=True).strip())))
''')
        name='deployment.json'
    else:
        assert args.label in sources and args.repetition
        s=sources[args.label]
        value=remote('SOURCE='+repr(s)+'\nLABEL='+repr(args.label)+'\nREP='+repr(args.repetition)+'\n'+'''
import json,os,subprocess,time
from pathlib import Path
repo=Path(SOURCE['repository'])
sha=SOURCE['source_git_commit']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==sha
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
assert subprocess.run(['pgrep','-f','[t]rain_model.py'],capture_output=True).returncode!=0
cache='/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface'
env=dict(os.environ,BETTERLVIT_EXPERIMENT=SOURCE['profile'],BETTERLVIT_SEED=str(SOURCE['seed']),
    BETTERLVIT_EPOCHS='80',BETTERLVIT_BATCH_SIZE='16',BETTERLVIT_GIT_COMMIT=sha,
    HF_HOME=cache,HF_HUB_CACHE=cache+'/hub',HF_MODULES_CACHE=cache+'/modules',
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED=str(SOURCE['seed']),
    TEST_SPLIT_ALLOWED='0',AUTO_TEST_EVALUATE='0',TOKENIZERS_PARALLELISM='false')
log=Path('/root/recipe_runs')/(LABEL+'_preflight_'+str(REP)+'.log')
assert not log.exists()
with log.open('w') as out:
    rc=subprocess.run(['/root/autodl-tmp/envs/betterlvit-paper/bin/python','tools/preflight_recipe.py'],
        cwd=repo,env=env,stdout=out,stderr=subprocess.STDOUT).returncode
assert rc==0,log.read_text()[-3000:]
value=json.loads(log.read_text().splitlines()[-1])
assert value['source_git_commit']==sha and value['seed']==SOURCE['seed']
value.update(repository=str(repo),repetition=REP,recorded_unix=time.time())
log.with_suffix('.json').write_text(json.dumps(value,indent=2)+'\\n')
print(json.dumps(value))
''',timeout=180)
        name=f'{args.label}_preflight_{args.repetition}.json'
    (ROOT/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(value))


if __name__=='__main__':main()

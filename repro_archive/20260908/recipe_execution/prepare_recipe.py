"""Deploy immutable recipe sources; execute bounded pre-training checks."""
import argparse
import json
import subprocess
from pathlib import Path

KEY = 'C:/Users/dtftn/.ssh/seetacloud_betterlvit_ed25519'
PYTHON = '/root/autodl-tmp/envs/betterlvit-paper/bin/python'
ROOT = Path('D:/BetterLViT/outputs/recipe_20260908')
HERE = Path(__file__).resolve().parent


def remote(code, timeout=60):
    result = subprocess.run(['ssh', '-i', KEY, '-p', '21465', '-o', 'BatchMode=yes',
        '-o', 'ConnectTimeout=15', 'root@connect.westb.seetacloud.com', PYTHON+' -'],
        input=code, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr+result.stdout[-4000:])
    return json.loads(result.stdout)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['deploy','update','checks','preflight'])
    p.add_argument('--sources', type=Path, default=HERE/'sources.json')
    p.add_argument('--bundle', type=Path)
    p.add_argument('--label', required=True)
    p.add_argument('--repetition', type=int, default=1)
    p.add_argument('--profile')
    args = p.parse_args()
    source = json.loads(args.sources.read_text())[args.label]
    header = 'SOURCE='+repr(source)+'\nLABEL='+repr(args.label)+'\nREP='+repr(args.repetition)+'\n'
    if args.action in ('deploy','update'):
        assert args.bundle
        subprocess.run(['scp','-i',KEY,'-P','21465','-o','BatchMode=yes',str(args.bundle),
            'root@connect.westb.seetacloud.com:/root/recipe_source.bundle'],check=True)
        value = remote(header+'ACTION='+repr(args.action)+'\n'+'''
import json, shutil, subprocess
from pathlib import Path
base=Path('/root/BetterLViT-visual-prior-dev')
repo=Path(SOURCE['repository'])
if ACTION=='deploy':
    assert not repo.exists()
    subprocess.run(['git','fetch','/root/recipe_source.bundle',SOURCE['branch']+':'+SOURCE['branch']],cwd=base,check=True,stdout=subprocess.PIPE)
    subprocess.run(['git','worktree','add',str(repo),SOURCE['branch']],cwd=base,check=True,stdout=subprocess.PIPE)
    (repo/'datasets').symlink_to('/root/autodl-tmp/datasets',target_is_directory=True)
else:
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
    assert not (Path('/root/recipe_runs')/(LABEL+'_80_20260908')).exists(), 'Never update a dispatched source'
    subprocess.run(['git','fetch','/root/recipe_source.bundle',SOURCE['branch']],cwd=repo,check=True,stdout=subprocess.PIPE)
    subprocess.run(['git','merge','--ff-only','FETCH_HEAD'],cwd=repo,check=True,stdout=subprocess.PIPE)
actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert actual==SOURCE['source_git_commit']
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
result=dict(source=SOURCE,system_free_bytes=shutil.disk_usage('/root').free,
    scratch_free_bytes=shutil.disk_usage('/root/autodl-tmp').free,
    fs_bytes=int(subprocess.check_output(['du','-sb','/autodl-fs/data'],text=True).split()[0]),
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total','--format=csv,noheader'],text=True).strip())
assert result['fs_bytes'] < 20_000_000_000
assert result['system_free_bytes'] > 5_000_000_000
Path('/root/recipe_runs').mkdir(exist_ok=True)
print(json.dumps(result))
''')
        name = args.label+('_deployment.json' if args.action=='deploy' else '_source_update.json')
    else:
        header += 'PROFILE='+repr(args.profile or source['profile'])+'\nACTION='+repr(args.action)+'\n'
        value = remote(header+'''
import json, os, subprocess, time
from pathlib import Path
repo=Path(SOURCE['repository'])
sha=SOURCE['source_git_commit']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==sha
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
assert subprocess.run(['pgrep','-f','[t]rain_model.py'],capture_output=True).returncode!=0
cache='/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface'
env=dict(os.environ, BETTERLVIT_EXPERIMENT=PROFILE,BETTERLVIT_SEED='1219',BETTERLVIT_EPOCHS='80',
    BETTERLVIT_BATCH_SIZE='16',BETTERLVIT_GIT_COMMIT=sha,
    HF_HOME=cache,HF_HUB_CACHE=cache+'/hub',HF_MODULES_CACHE=cache+'/modules',
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='1219',
    TEST_SPLIT_ALLOWED='0',AUTO_TEST_EVALUATE='0',TOKENIZERS_PARALLELISM='false')
log=Path('/root/recipe_runs')/(LABEL+'_'+ACTION+'_'+str(REP)+'.log')
assert not log.exists(), 'Check already recorded'
commands=([['tools/check_training_recipe.py'],['tools/test_visual_validation_gate.py']]
    if ACTION=='checks' else [['tools/preflight_recipe.py']])
records=[]
with log.open('w') as out:
    for command in commands:
        started=time.time()
        rc=subprocess.run(['/root/autodl-tmp/envs/betterlvit-paper/bin/python']+command,cwd=repo,env=env,stdout=out,stderr=subprocess.STDOUT).returncode
        records.append(dict(command=command,returncode=rc,seconds=time.time()-started))
        if rc:break
if rc:
    print(json.dumps(dict(status='failed',records=records,log=log.read_text(),source=SOURCE)))
else:
    value=(dict(status='ok',records=records,log=log.read_text()) if ACTION=='checks'
        else json.loads(log.read_text().splitlines()[-1]))
    value.update(source_git_commit=sha,repository=str(repo),repetition=REP,recorded_unix=time.time())
    log.with_suffix('.json').write_text(json.dumps(value,indent=2)+'\\n')
    print(json.dumps(value))
''', timeout=240)
        name = f'{args.label}_{args.action}_{args.repetition}.json'
    (ROOT/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(value))
    assert value.get('status') != 'failed', 'Pre-training verification failed; inspect saved output'


if __name__ == '__main__':
    main()

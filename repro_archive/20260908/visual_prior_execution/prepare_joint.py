"""Deploy registered immutable sources or run one temporary GPU preflight."""
import argparse
import json
import subprocess
from pathlib import Path

KEY = 'C:/Users/dtftn/.ssh/seetacloud_betterlvit_ed25519'
PYTHON = '/root/autodl-tmp/envs/betterlvit-paper/bin/python'
SOURCES = {
    'p12': ['paper/p12-frozen-visual-prior', '5d09913d46863073cba93159af4ed61f88fdd96e', 'p12_visual_prior'],
    'c9': ['paper/c9-frozen-visual-random', '3d60890f61e7ef9b1ee4560d38436c6532ba6fd0', 'c9_visual_random'],
}


def remote(code, timeout=60):
    result = subprocess.run(['ssh', '-i', KEY, '-p', '21465', '-o', 'BatchMode=yes',
        '-o', 'ConnectTimeout=15', 'root@connect.westb.seetacloud.com', PYTHON + ' -'],
        input=code, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr + result.stdout[-3000:])
    return json.loads(result.stdout)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['deploy', 'preflight'])
    p.add_argument('--label', choices=list(SOURCES))
    p.add_argument('--repetition', type=int, choices=[1, 2])
    args = p.parse_args()
    header = 'SOURCES=' + repr(SOURCES) + '\n'
    if args.action == 'deploy':
        subprocess.run(['scp', '-i', KEY, '-P', '21465', '-o', 'BatchMode=yes',
            'D:/BetterLViT/outputs/visual_prior_formal.bundle',
            'root@connect.westb.seetacloud.com:/root/visual_prior_formal.bundle'], check=True)
        value = remote(header + '''
import json, shutil, subprocess
from pathlib import Path
base=Path('/root/BetterLViT-visual-prior-dev')
result={}
for label,(branch,sha,profile) in SOURCES.items():
    repo=Path('/root/BetterLViT-visual-'+label)
    assert not repo.exists(), 'Worktree already exists'
    subprocess.run(['git','fetch','/root/visual_prior_formal.bundle',branch+':'+branch],cwd=base,check=True,stdout=subprocess.PIPE)
    subprocess.run(['git','worktree','add',str(repo),branch],cwd=base,check=True,stdout=subprocess.PIPE)
    (repo/'datasets').symlink_to('/root/autodl-tmp/datasets',target_is_directory=True)
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    assert actual==sha
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
    assert json.loads((repo/'experiment_manifests/active_visual.json').read_text())['profile']==profile
    result[label]={'source_git_commit':actual,'repository':str(repo)}
result['system_free_bytes']=shutil.disk_usage('/root').free
result['scratch_free_bytes']=shutil.disk_usage('/root/autodl-tmp').free
result['fs_bytes']=int(subprocess.check_output(['du','-sb','/autodl-fs/data'],text=True).split()[0])
assert result['fs_bytes'] < 20_000_000_000
assert result['system_free_bytes'] > 4_000_000_000
print(json.dumps(result))
''')
        name = 'formal_deployment.json'
    else:
        assert args.label and args.repetition
        value = remote(header + f'LABEL={args.label!r}\nREP={args.repetition!r}\n' + '''
import json, os, subprocess, time
from pathlib import Path
branch,sha,profile=SOURCES[LABEL]
repo=Path('/root/BetterLViT-visual-'+LABEL)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==sha
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
assert subprocess.run(['pgrep','-f','[t]rain_model.py'],capture_output=True).returncode!=0
log=Path('/root/visual_prior_runs')/(LABEL+'_frozen_preflight_'+str(REP)+'.log')
assert not log.exists(), 'Preflight already recorded'
cache='/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface'
env=dict(os.environ, BETTERLVIT_EXPERIMENT=profile,BETTERLVIT_SEED='1219',BETTERLVIT_EPOCHS='80',
    BETTERLVIT_BATCH_SIZE='16',BETTERLVIT_GIT_COMMIT=sha,BETTERLVIT_VISUAL_MODEL_ROOT='/root/visual_prior_models',
    HF_HOME=cache,HF_HUB_CACHE=cache+'/hub',HF_MODULES_CACHE=cache+'/modules',
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='1219',
    TEST_SPLIT_ALLOWED='0',AUTO_TEST_EVALUATE='0',TOKENIZERS_PARALLELISM='false')
with log.open('w') as out:
    rc=subprocess.run(['/root/autodl-tmp/envs/betterlvit-paper/bin/python','tools/preflight_visual_joint.py'],
        cwd=repo,env=env,stdout=out,stderr=subprocess.STDOUT).returncode
assert rc==0, log.read_text()[-2500:]
value=json.loads(log.read_text().splitlines()[-1])
value.update(source_git_commit=sha,repository=str(repo),repetition=REP,recorded_unix=time.time())
log.with_suffix('.json').write_text(json.dumps(value,indent=2)+'\\n')
print(json.dumps(value))
''')
        name = f'{args.label}_frozen_preflight_{args.repetition}.json'
    output = Path('D:/BetterLViT/outputs/visual_prior_20260908') / name
    output.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(value))


if __name__ == '__main__':
    main()

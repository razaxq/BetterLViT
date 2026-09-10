"""Deploy immutable Git bundles and run bounded pre-training checks."""
import argparse
import json
from pathlib import Path
import subprocess
from remote_ops import HERE,KEY,remote,save


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('deploy','update','checks','legacy','preflight'))
    p.add_argument('--label',default='dev');p.add_argument('--profile');p.add_argument('--name',required=True)
    p.add_argument('--bundle');p.add_argument('--audit',action='store_true');p.add_argument('--disable-aux-loss',action='store_true')
    args=p.parse_args();source=json.loads((HERE/'sources.json').read_text())[args.label]
    out=HERE/'preflight'/(args.name+'.json')
    assert not out.exists()
    header='SOURCE='+repr(source)+'\nARGS='+repr(vars(args))+'\n'
    if args.action in ('deploy','update'):
        assert args.bundle
        subprocess.run(['scp','-i',KEY,'-P','21465','-o','BatchMode=yes',args.bundle,
            'root@connect.westb.seetacloud.com:/root/autodl-tmp/visual_aux_source.bundle'],check=True)
        value=remote(header+'''
import json,shutil,subprocess
from pathlib import Path
base=Path('/root/BetterLViT-visual-prior-dev');repo=Path(SOURCE['repository'])
bundle='/root/autodl-tmp/visual_aux_source.bundle'
if ARGS['action']=='deploy':
    assert not repo.exists()
    subprocess.run(['git','fetch',bundle,SOURCE['branch']+':'+SOURCE['branch']],cwd=base,check=True,stdout=subprocess.PIPE)
    subprocess.run(['git','worktree','add',str(repo),SOURCE['branch']],cwd=base,check=True,stdout=subprocess.PIPE)
    (repo/'datasets').symlink_to('/root/autodl-tmp/datasets',target_is_directory=True)
else:
    # The initial deployment exposed historical CRLF blobs under a new LF
    # attribute. Permit only this verified line-ending-only development repair.
    subprocess.run(['git','diff','--ignore-space-at-eol','--exit-code'],cwd=repo,check=True,stdout=subprocess.PIPE)
    assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=repo,text=True).strip()
    assert not (repo/'Covid19/BetterLViT'/SOURCE['profile']).exists()
    subprocess.run(['git','fetch',bundle,SOURCE['branch']],cwd=repo,check=True,stdout=subprocess.PIPE)
    subprocess.run(['git','merge','--ff-only','FETCH_HEAD'],cwd=repo,check=True,stdout=subprocess.PIPE)
sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert sha==SOURCE['source_git_commit']
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
Path('/root/autodl-tmp/visual_aux_preflight').mkdir(exist_ok=True)
Path('/root/autodl-tmp/visual_aux_runs').mkdir(exist_ok=True)
print(json.dumps(dict(source=SOURCE,status='ok',scratch_free_bytes=shutil.disk_usage(repo).free)))
''')
    else:
        value=remote(header+'''
import json,os,subprocess,time
from pathlib import Path
repo=Path(SOURCE['repository']);sha=SOURCE['source_git_commit'];profile=ARGS['profile'] or SOURCE['profile']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==sha
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
assert subprocess.run(['pgrep','-f','[t]rain_model.py'],capture_output=True).returncode==1
cache='/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface'
env=dict(os.environ,BETTERLVIT_EXPERIMENT=profile,BETTERLVIT_SEED='1219',BETTERLVIT_EPOCHS='80',
    BETTERLVIT_BATCH_SIZE='16',BETTERLVIT_GIT_COMMIT=sha,HF_HOME=cache,HF_HUB_CACHE=cache+'/hub',
    HF_MODULES_CACHE=cache+'/modules',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
    CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='1219',TOKENIZERS_PARALLELISM='false',
    TEST_SPLIT_ALLOWED='0',AUTO_TEST_EVALUATE='0')
log=Path('/root/autodl-tmp/visual_aux_preflight')/(ARGS['name']+'.log');assert not log.exists()
script={'checks':'tools/test_visual_aux.py','legacy':'tools/preflight_recipe.py','preflight':'tools/preflight_visual_aux.py'}[ARGS['action']]
command=['/root/autodl-tmp/envs/betterlvit-paper/bin/python',script]
if ARGS['audit']:command+=['--audit-dir',str(log.with_suffix(''))]
if ARGS['disable_aux_loss']:command+=['--disable-aux-loss']
started=time.time()
with log.open('w') as out:
    rc=subprocess.run(command,cwd=repo,env=env,stdout=out,stderr=subprocess.STDOUT).returncode
value=json.loads(log.read_text().splitlines()[-1]) if rc==0 and ARGS['action']!='checks' else dict(status='ok' if rc==0 else 'failed',log=log.read_text())
value.update(source_git_commit=sha,profile=profile,command=command,seconds=time.time()-started,returncode=rc)
log.with_suffix('.json').write_text(json.dumps(value,indent=2)+'\\n')
print(json.dumps(value))
''',timeout=300)
    save('preflight/'+args.name+'.json',value)
    print(json.dumps({k:v for k,v in value.items() if k not in ('training_recipe','output_sha256_each_step')},indent=2))
    assert value['status']=='ok'


if __name__=='__main__':main()

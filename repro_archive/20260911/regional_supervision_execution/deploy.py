"""Deploy immutable git bundles into isolated remote repositories."""
import argparse,json,subprocess
from pathlib import Path
from remote_ops import HERE,remote,save,copy_to_remote


def deploy(label,local):
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=local,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=local,text=True).strip()
    bundle=Path('D:/BetterLViT/outputs')/(label+'_'+sha[:8]+'.bundle')
    subprocess.run(['git','bundle','create',str(bundle),'9eca26de5b301099805530edbf5a1a8718bea662..HEAD'],cwd=local,check=True,capture_output=True)
    remote_bundle='/root/autodl-tmp/'+bundle.name
    copy_to_remote(bundle,remote_bundle)
    repo='/root/autodl-tmp/BetterLViT-regional-'+label
    value=remote('REPO='+repr(repo)+'\nBUNDLE='+repr(remote_bundle)+'\nSHA='+repr(sha)+'\n'+'''
import json,subprocess,shutil
from pathlib import Path
repo=Path(REPO)
if not repo.exists():
    subprocess.run(['git','clone','--shared','--no-checkout','/root/BetterLViT-recipe-r2',str(repo)],check=True,capture_output=True)
else:
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
    assert not list((repo/'Covid19').glob('**/*.pth.tar')),'Never update a trained worktree'
subprocess.run(['git','fetch',BUNDLE,'HEAD'],cwd=repo,check=True,capture_output=True)
subprocess.run(['git','checkout','--detach',SHA],cwd=repo,check=True,capture_output=True)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==SHA
if not (repo/'datasets').exists():(repo/'datasets').symlink_to('/root/autodl-tmp/datasets')
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
print(json.dumps(dict(repository=str(repo),source_git_commit=SHA,clean=True,root_free=shutil.disk_usage('/root').free,scratch_free=shutil.disk_usage(repo).free)))
''')
    save(label+'_deployment.json',value)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--local',required=True)
    args=p.parse_args();print(json.dumps(deploy(args.label,args.local),indent=2))

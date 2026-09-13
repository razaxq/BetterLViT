"""Deploy frozen sources, then push and independently resolve GitHub refs."""
import json,subprocess
from pathlib import Path
from remote_ops import HERE,read,remote,save,copy_to_remote

def main():
    sources=read(HERE/'sources.json')
    for label,s in sources.items():
        repo=s['local_repository'];sha=s['source_git_commit']
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==sha
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
        bundle=Path('D:/BetterLViT/outputs')/(label+'_'+sha[:8]+'.bundle')
        subprocess.run(['git','bundle','create',str(bundle),'9eca26de5b301099805530edbf5a1a8718bea662..HEAD'],cwd=repo,check=True,capture_output=True)
        rb='/root/autodl-tmp/'+bundle.name;copy_to_remote(bundle,rb)
        value=remote('SOURCE='+repr(s)+'\nBUNDLE='+repr(rb)+'\n'+'''
import json,subprocess,shutil
from pathlib import Path
repo=Path(SOURCE['repository'])
if not repo.exists():
    subprocess.run(['git','clone','--shared','--no-checkout','/root/BetterLViT-recipe-r2',str(repo)],check=True,capture_output=True)
else:
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
    assert not list((repo/'Covid19').glob('**/*.pth.tar')),'Never update a trained worktree'
subprocess.run(['git','fetch',BUNDLE,'HEAD'],cwd=repo,check=True,capture_output=True)
subprocess.run(['git','checkout','--detach',SOURCE['source_git_commit']],cwd=repo,check=True,capture_output=True)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==SOURCE['source_git_commit']
if not (repo/'datasets').exists():(repo/'datasets').symlink_to('/root/autodl-tmp/datasets')
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
print(json.dumps(dict(repository=str(repo),source_git_commit=SOURCE['source_git_commit'],clean=True,scratch_free_bytes=shutil.disk_usage(repo).free)))
''')
        save(label+'_deployment.json',value);print(json.dumps(value),flush=True)
    github='https://github.com/razaxq/BetterLViT.git';repo=next(iter(sources.values()))['local_repository']
    refs=[ref for s in sources.values() for ref in ('refs/heads/'+s['branch'],'refs/tags/'+s['experiment_tag'])]
    subprocess.run(['git','push',github,*refs],cwd=repo,check=True,capture_output=True)
    raw=subprocess.check_output(['git','ls-remote',github,*refs],cwd=repo,text=True)
    resolved={line.split()[1]:line.split()[0] for line in raw.splitlines()}
    for s in sources.values():
        for ref in ('refs/heads/'+s['branch'],'refs/tags/'+s['experiment_tag']):assert resolved[ref]==s['source_git_commit']
    save('github_sources_verified.json',dict(verified=True,repository=github,refs=resolved))
    print(json.dumps(dict(github_verified=True,refs=resolved)))

if __name__=='__main__':main()

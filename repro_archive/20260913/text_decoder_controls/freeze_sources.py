"""Freeze distinct manifests/commits/tags from shared checked-in implementation."""
import json
from pathlib import Path
import subprocess
from remote_ops import save
dev=Path('D:/BetterLViT/text_decoder_work')
def git(*args,cwd=dev):
    return subprocess.check_output(['git',*args],cwd=cwd,text=True).strip()
assert not git('status','--porcelain')
parent=git('rev-parse','HEAD')
base=json.loads((dev/'experiment_manifests/active_recipe.json').read_text())
sources={}
for label,mode in [('t1','visual'),('t2','text')]:
    folder=Path('D:/BetterLViT/text_'+label+'_work')
    branch='paper/'+label+'-decoder-80e-seed1219'
    tag='experiment-'+label+'-decoder-80e-seed1219-20260913'
    assert not folder.exists()
    git('worktree','add','-b',branch,str(folder),parent)
    m=dict(base,profile=label+'_decoder_'+mode,paper_id=label.upper(),decoder_context_mode=mode,
        adapter_parameters=16896,control_profile='r2_single_cosine',
        control_source_git_commit='9eca26de5b301099805530edbf5a1a8718bea662',
        development_source_git_commit=parent,experiment_tag=tag,
        plan='docs/TEXT_DECODER_CONTROLS_20260913.md')
    (folder/'experiment_manifests/active_decoder.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8',newline='\n')
    git('add','experiment_manifests/active_decoder.json',cwd=folder)
    git('commit','-m','Register '+label.upper()+' matched decoder '+mode+' control at 80 epochs',cwd=folder)
    sha=git('rev-parse','HEAD',cwd=folder)
    git('tag',tag,sha,cwd=folder)
    sources[label]=dict(source_git_commit=sha,development_source_git_commit=parent,
        branch=branch,experiment_tag=tag,local_repository=str(folder),
        repository='/root/autodl-tmp/BetterLViT-text-'+label,
        remote_run='/root/autodl-tmp/text_decoder_runs/'+label+'_80e_20260913',
        profile=m['profile'],seed=1219,previous_label=None if label=='t1' else 't1')
save('sources.json',sources)
print(json.dumps(sources,indent=2))

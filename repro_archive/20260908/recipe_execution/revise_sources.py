"""Append the platform-rounding validation fix; never rewrite original tags."""
import json
import subprocess
from pathlib import Path
from prepare_recipe import HERE, ROOT

sources=json.loads((HERE/'sources.json').read_text())
(HERE/'source_registration_v1.json').write_text(json.dumps(sources,indent=2)+'\n',encoding='utf-8')
fix='3c10d261e8509da4d96a97c183d75aa231427fd1'
for key in ('r1','r2'):
    value=sources[key]
    subprocess.run(['git','cherry-pick',fix],cwd=value['local_repository'],check=True)
    value['previous_unlaunched_source_git_commit']=value['source_git_commit']
    value['previous_unlaunched_tag']=value['experiment_tag']
    value['source_git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=value['local_repository'],text=True).strip()
    value['experiment_tag']+='-v2'
    value['preflight_repetitions']=[3,4]
    # The original manifest tag is retained as a protocol registration reference.
    # The v2 tag identifies the final runtime source containing the validator fix.
    subprocess.run(['git','tag','-a',value['experiment_tag'],'-m','Same registered protocol; cross-platform LR validation fix',value['source_git_commit']],cwd=value['local_repository'],check=True)
sources['dev']['initial_source_git_commit']=sources['dev']['source_git_commit']
sources['dev']['source_git_commit']=fix
(HERE/'sources.json').write_text(json.dumps(sources,indent=2)+'\n',encoding='utf-8')
dev=Path('D:/BetterLViT/recipe_work')
subprocess.run(['git','bundle','create','D:/BetterLViT/outputs/recipe_formal_v2.bundle',*[s['branch'] for s in sources.values()]],cwd=dev,check=True)
subprocess.run(['git','push','https://github.com/razaxq/BetterLViT.git',*[s['branch'] for s in sources.values()],*[sources[k]['experiment_tag'] for k in ('r1','r2')]],cwd=dev,check=True)
print(json.dumps(sources,indent=2))

"""Freeze the four predeclared additional-seed runs without altering training math."""
import copy
import json
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path('D:/BetterLViT/outputs/recipe_replication_20260909')
DEV=Path('D:/BetterLViT/recipe_replication_work')
BASE='bb03754c783de6ef6c202a0c58f2bd63f2b6ff6b'
sys.path.insert(0,str(DEV))
from training_recipe import planned_rates,validate_recipe_manifest


def git(*args,cwd=DEV):
    return subprocess.check_output(['git',*args],cwd=cwd,text=True).strip()


assert git('rev-parse','HEAD')==BASE
assert not git('diff','3c10d261e8509da4d96a97c183d75aa231427fd1','--','train_model.py','Load_Dataset.py','Config.py','paper_experiments.py','nets','Train_one_epoch.py','utils.py')
template=json.loads(Path('D:/BetterLViT/outputs/recipe_20260908/r2_manifest.json').read_text())
sources={}
previous=None
for seed in (2027,3407):
    for role in ('c4','r2'):
        label=role+'s'+str(seed)
        work=Path('D:/BetterLViT/recipe_'+label+'_work')
        branch='paper/'+label+'-replication'
        tag='experiment-'+role+'-recipe-80e-seed'+str(seed)+'-20260909'
        assert not work.exists()
        git('worktree','add','-b',branch,str(work),BASE)
        manifest=copy.deepcopy(template)
        manifest.update(profile='c4_race_pe_control' if role=='c4' else 'r2_single_cosine',
            paper_id=role.upper(),seed=seed,augmentation_policy='legacy',
            lr_schedule='warm_restarts' if role=='c4' else 'single_cosine',
            development_source_git_commit=BASE,experiment_tag=tag,
            plan='docs/RECIPE_REPLICATION_PLAN_20260909.md',replication_cohort='r2_additional_seeds_v1',
            paired_role=role)
        manifest['planned_epoch_lrs']=planned_rates(manifest['lr_schedule'])
        manifest['control_source_git_commit']=(sources['c4s'+str(seed)]['source_git_commit'] if role=='r2'
            else 'add4908a0d6f702b0a10c4581725b535543829b8')
        validate_recipe_manifest(manifest)
        invalid=dict(manifest,seed=42)
        try:validate_recipe_manifest(invalid)
        except AssertionError:pass
        else:raise AssertionError('Unregistered seed was accepted')
        (work/'experiment_manifests/active_recipe.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
        git('add','experiment_manifests/active_recipe.json',cwd=work)
        git('commit','-m','Freeze '+label+' matched 80-epoch replication',cwd=work)
        sha=git('rev-parse','HEAD',cwd=work)
        git('tag','-a',tag,'-m','Predeclared matched R2 recipe replication',sha)
        sources[label]=dict(branch=branch,source_git_commit=sha,profile=manifest['profile'],seed=seed,role=role,
            previous_label=previous,experiment_tag=tag,repository='/root/BetterLViT-recipe-'+label,
            local_repository=str(work),remote_run='/root/recipe_runs/'+label+'_80_20260909')
        (ROOT/f'{label}_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
        previous=label
(HERE/'sources.json').write_text(json.dumps(sources,indent=2)+'\n',encoding='utf-8',newline='\n')
git('bundle','create','D:/BetterLViT/outputs/recipe_replication.bundle','paper/recipe-replication-development',*[v['branch'] for v in sources.values()])
git('push','https://github.com/razaxq/BetterLViT.git','paper/recipe-replication-development',*[v['branch'] for v in sources.values()],*[v['experiment_tag'] for v in sources.values()])
print(json.dumps(sources,indent=2))

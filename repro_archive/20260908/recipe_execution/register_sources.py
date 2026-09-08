"""Create two independently frozen formal sources before their GPU preflights."""
import json
import subprocess
import sys
from pathlib import Path
from prepare_recipe import HERE, ROOT

dev = Path('D:/BetterLViT/recipe_work')
sys.path.insert(0,str(dev))
from training_recipe import planned_rates, validate_recipe_manifest


def git(*args, cwd=dev):
    return subprocess.check_output(['git',*args],cwd=cwd,text=True).strip()


sources = json.loads((HERE/'sources.json').read_text())
assert git('rev-parse','HEAD') == sources['dev']['source_git_commit']
assert not git('status','--porcelain','--untracked-files=no')
for label,profile,augmentation,schedule in (
    ('r1','r1_chest_augmentation','chest_orientation','warm_restarts'),
    ('r2','r2_single_cosine','legacy','single_cosine'),
):
    work = Path('D:/BetterLViT/recipe_'+label+'_work')
    branch = 'paper/'+label+'-training-recipe'
    tag = 'experiment-'+label+'-recipe-80e-seed1219-20260908'
    assert not work.exists()
    git('worktree','add','-b',branch,str(work),sources['dev']['source_git_commit'])
    manifest = dict(profile=profile,paper_id=label.upper(),seed=1219,epochs=80,
        batch_size=16,image_size=224,train_drop_last=True,num_workers=4,
        initialization='from_scratch',loss_name='dice_focal',optimizer='Adam',weight_decay=1e-4,
        selection_metric='iou',threshold=.5,augmentation_policy=augmentation,lr_schedule=schedule,
        planned_epoch_lrs=planned_rates(schedule),lora=False,boundary_loss=False,
        new_supervision=False,visual_prior=False,test_split_allowed=False,auto_test_evaluate=False,
        auto_validation_export=True,control_profile='c4_race_pe_control',
        control_source_git_commit='add4908a0d6f702b0a10c4581725b535543829b8',
        development_source_git_commit=sources['dev']['source_git_commit'],
        experiment_tag=tag,plan='docs/TRAINING_RECIPE_PLAN_20260908.md')
    validate_recipe_manifest(manifest)
    (work/'experiment_manifests/active_recipe.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    git('add','experiment_manifests/active_recipe.json',cwd=work)
    git('commit','-m','Freeze '+label.upper()+' 80-epoch seed1219 recipe protocol',cwd=work)
    sha = git('rev-parse','HEAD',cwd=work)
    git('tag','-a',tag,'-m','Registered '+label.upper()+' recipe; 80 epochs; Val screening only',sha)
    sources[label] = dict(branch=branch,source_git_commit=sha,repository='/root/BetterLViT-recipe-'+label,
        local_repository=str(work),profile=profile,experiment_tag=tag,
        remote_run='/root/recipe_runs/'+label+'_80_20260908')
    (ROOT/(label+'_manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
(HERE/'sources.json').write_text(json.dumps(sources,indent=2)+'\n',encoding='utf-8')
git('bundle','create','D:/BetterLViT/outputs/recipe_formal.bundle',sources['r1']['branch'],sources['r2']['branch'])
git('push','https://github.com/razaxq/BetterLViT.git',*[sources[k]['branch'] for k in ('r1','r2')],*[sources[k]['experiment_tag'] for k in ('r1','r2')])
print(json.dumps(sources,indent=2))

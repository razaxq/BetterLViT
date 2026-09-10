"""Give S1/S2 distinct immutable commits, tags and manifests before training."""
import json
from pathlib import Path
import subprocess
import sys
from remote_ops import HERE,save
sys.path.insert(0,'D:/BetterLViT/visual_aux_work')
from training_recipe import planned_rates
from visual_aux_protocol import validate_visual_aux_manifest


def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()


def main():
    sources=json.loads((HERE/'sources.json').read_text());dev=sources['dev']
    assert git(dev['local_repository'],'rev-parse','HEAD')==dev['source_git_commit']
    assert not git(dev['local_repository'],'status','--porcelain')
    checks=json.loads((HERE/'preflight/dev_checks.json').read_text())
    assert checks['status']=='ok'
    original=json.loads((HERE.parents[1]/'20260908/recipe_execution/r2_preflight_3.json').read_text())
    legacy=json.loads((HERE/'preflight/dev_legacy_r2.json').read_text())
    for k in ('initial_base_sha256','input_image_sha256','output_sha256_each_step','loss_each_step'):
        assert original[k]==legacy[k],k
    for label,mode,profile in [('s1','pixel','s1_r2_pixel_aux'),('s2','regional','s2_r2_visual_aux')]:
        assert label not in sources
        repo=Path('D:/BetterLViT')/('visual_aux_'+label+'_work')
        assert not repo.exists()
        branch='paper/r2-visual-aux-'+label
        git(dev['local_repository'],'worktree','add',str(repo),'-b',branch,dev['source_git_commit'])
        manifest=dict(profile=profile,seed=1219,visual_aux_mode=mode,epochs=80,batch_size=16,
            initialization='from_scratch',selection_metric='iou',threshold=.5,augmentation_policy='legacy',
            lr_schedule='single_cosine',planned_epoch_lrs=planned_rates('single_cosine'),loss_name='dice_focal',
            new_supervision=True,test_split_allowed=False,auto_test_evaluate=False,lora=False,boundary_loss=False,
            visual_prior=False,text_auxiliary=False,inference_routing=False,train_audit_epochs=[20,40,60,80],
            effective_aux_coefficients=dict(pixel=.02,presence=.01 if mode=='regional' else 0.,
                                             occupancy=.005 if mode=='regional' else 0.),
            baseline_source_git_commit='9eca26de5b301099805530edbf5a1a8718bea662',
            protocol='docs/VISUAL_AUX_PLAN_20260910.md')
        validate_visual_aux_manifest(manifest)
        path=repo/'experiment_manifests/active_visual_aux.json';path.write_text(json.dumps(manifest,indent=2)+'\n',newline='\n')
        git(repo,'add','experiment_manifests/active_visual_aux.json')
        git(repo,'commit','-m','Freeze '+label.upper()+' R2 visual supervision experiment')
        sha=git(repo,'rev-parse','HEAD');tag='experiment-'+label+'-r2-'+mode+'-80e-seed1219-20260910'
        git(repo,'tag','-a',tag,'-m','Frozen '+label.upper()+' visual auxiliary experiment')
        sources[label]=dict(branch=branch,source_git_commit=sha,profile=profile,local_repository=str(repo),
            repository='/root/autodl-tmp/BetterLViT-visual-aux-'+label,
            remote_run='/root/autodl-tmp/visual_aux_runs/'+label+'_80_seed1219_20260910',experiment_tag=tag)
        save('sources.json',sources);save(label+'_manifest.json',manifest)
        git(repo,'push','https://github.com/razaxq/BetterLViT.git','HEAD:'+branch,'refs/tags/'+tag)
    save('baseline_reuse_verified.json',dict(original_training_sha=original['source_git_commit'],
        preflight_source_sha=legacy['source_git_commit'],shared_initialization_and_legacy_5_steps_identical=True,
        matched_fields=['initial_base_sha256','input_image_sha256','output_sha256_each_step','loss_each_step'],
        baseline_result='../../20260908/recipe_execution/r2_results/validation.json'))
    print(json.dumps(sources,indent=2))


if __name__=='__main__':main()

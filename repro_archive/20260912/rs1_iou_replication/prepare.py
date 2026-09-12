"""Make separately pinned RS1 seed replicas; unchanged training and network code."""
import hashlib
import json
from pathlib import Path
import subprocess
from remote_ops import HERE,DOCS,read,save
BASE='b4dd566ae472079c55e41cffd7727060bcfd6bf5'
ORIGINAL=Path('D:/BetterLViT/regional_rs1_work')
def git(repo,*args):return subprocess.check_output(['git',*args],cwd=repo,text=True).strip()
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    assert not (HERE/'sources.json').exists()
    assert git(ORIGINAL,'rev-parse','HEAD')==BASE and not git(ORIGINAL,'status','--porcelain')
    plan_sha=git(DOCS,'rev-parse','HEAD')
    assert subprocess.check_output(['git','show',plan_sha+':'+(HERE/'PROTOCOL.md').relative_to(DOCS).as_posix()],cwd=DOCS)==(HERE/'PROTOCOL.md').read_bytes()
    controls=read(DOCS/'repro_archive/20260909/recipe_replication/sources.json');sources={}
    for seed in (2027,3407):
        label=f'rs1s{seed}';local=Path(f'D:/BetterLViT/rs1_iou_s{seed}_work');branch=f'paper/rs1-iou-replication-seed{seed}'
        assert not local.exists()
        subprocess.run(['git','worktree','add','-b',branch,str(local),BASE],cwd=ORIGINAL,check=True,capture_output=True)
        p=local/'regional_protocol.py';old=p.read_text(encoding='utf-8');assert "m['seed'] == 1219" in old
        p.write_text(old.replace("m['seed'] == 1219","m['seed'] in (1219, 2027, 3407)"),encoding='utf-8',newline='\n')
        control=controls[f'r2s{seed}'];tag=f'experiment-rs1-iou-80e-seed{seed}-20260912'
        for name in ('active_recipe.json','active_regional.json'):
            path=local/'experiment_manifests'/name;m=read(path);m['seed']=seed
            if name=='active_regional.json':
                m.update(experiment_tag=tag,replication_of_source_git_commit=BASE,
                    control_source_git_commit=control['source_git_commit'],replication_plan_git_commit=plan_sha,
                    replication_plan_path=(HERE/'PROTOCOL.md').relative_to(DOCS).as_posix(),
                    primary_metric='per_image_macro_iou',secondary_metrics_are_not_strict_vetoes=True)
            path.write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8',newline='\n')
        (local/'docs/RS1_IOU_REPLICATION.md').write_bytes((HERE/'PROTOCOL.md').read_bytes())
        git(local,'add','--','regional_protocol.py','experiment_manifests/active_regional.json','experiment_manifests/active_recipe.json','docs/RS1_IOU_REPLICATION.md')
        git(local,'commit','-m',f'Freeze unchanged RS1 recipe replication for seed {seed} with IoU priority')
        sha=git(local,'rev-parse','HEAD');git(local,'tag',tag)
        changed=git(local,'diff','--name-only',BASE,sha).splitlines()
        assert set(changed)=={'regional_protocol.py','experiment_manifests/active_recipe.json','experiment_manifests/active_regional.json','docs/RS1_IOU_REPLICATION.md'}
        baseline=DOCS/f'repro_archive/20260909/recipe_replication/r2s{seed}_results/validation.json'
        v=read(baseline);assert v['seed']==seed and v['samples']==1429 and v['checkpoint_git_commit']==control['source_git_commit']
        source=dict(seed=seed,label=label,source_git_commit=sha,experiment_tag=tag,branch=branch,local_repository=str(local),
            repository=f'/root/autodl-tmp/BetterLViT-{label}-iou',remote_run=f'/root/autodl-tmp/rs1_iou_runs/{label}_20260912',
            profile='rs1_global_iou',replication_parent=BASE,baseline=control,
            baseline_validation_relative=baseline.relative_to(DOCS).as_posix(),baseline_validation_sha256=digest(baseline),
            changed_files_from_rs1=changed,previous_label=None if seed==2027 else 'rs1s2027')
        save(label+'_manifest.json',read(local/'experiment_manifests/active_regional.json'));sources[label]=source
    save('sources.json',sources);print(json.dumps(sources,ensure_ascii=False))
if __name__=='__main__':main()

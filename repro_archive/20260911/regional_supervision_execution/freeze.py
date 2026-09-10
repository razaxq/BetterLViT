"""Create three separately committed/tagged manifests from one calibrated codebase."""
import hashlib,json,subprocess
from pathlib import Path
from remote_ops import HERE,save


def git(repo,*args):
    return subprocess.check_output(['git',*args],cwd=repo,text=True).strip()


if __name__=='__main__':
    dev=Path('D:/BetterLViT/regional_supervision_work')
    assert not git(dev,'status','--porcelain')
    calibration=json.loads((HERE/'preflight/calibration.json').read_text())
    assert calibration['common_weight']==.128312
    development=git(dev,'rev-parse','HEAD')
    base=json.loads((dev/'experiment_manifests/active_recipe.json').read_text())
    sources={}
    for label,mode in (('rs1','global'),('rs2','local'),('rs3','balanced')):
        local=Path('D:/BetterLViT')/('regional_'+label+'_work')
        branch='paper/regional-'+label+'-80e-seed1219'
        assert not local.exists()
        subprocess.run(['git','worktree','add','-b',branch,str(local),development],cwd=dev,check=True,capture_output=True)
        profile={'rs1':'rs1_global_iou','rs2':'rs2_local_iou','rs3':'rs3_balanced_iou'}[label]
        tag='experiment-'+label+'-regional-80e-seed1219-20260911'
        manifest=dict(base,profile=profile,paper_id=label.upper(),new_supervision=True,
            control_profile='r2_single_cosine',control_source_git_commit='9eca26de5b301099805530edbf5a1a8718bea662',
            development_source_git_commit=development,experiment_tag=tag,regional_mode=mode,
            regional_weight=calibration['common_weight'],window=56,stride=28,eps=1e-6,
            empty_rule='mean_foreground_probability',extra_parameters=0,calibration_frozen=True,
            calibration_source_git_commit=calibration['source_git_commit'],
            calibration_json_sha256=hashlib.sha256((HERE/'preflight/calibration.json').read_bytes()).hexdigest(),
            audit_epochs=[20,40,60,80],plan='docs/REGIONAL_CALIBRATION.md',
            experiment_plan_git_commit='709e4ecbe7d34d3ca01b6a48bdc6259b82026d10',
            experiment_plan_path='docs/REGIONAL_SUPERVISION_EXPERIMENT_PLAN_20260911.md')
        path=local/'experiment_manifests/active_regional.json'
        path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        git(local,'add','--','experiment_manifests/active_regional.json')
        git(local,'commit','-m','Freeze '+label.upper()+' regional overlap 80-epoch experiment')
        sha=git(local,'rev-parse','HEAD');git(local,'tag',tag)
        sources[label]=dict(source_git_commit=sha,experiment_tag=tag,profile=profile,local_repository=str(local),
            branch=branch,repository='/root/autodl-tmp/BetterLViT-regional-'+label,
            remote_run='/root/autodl-tmp/regional_runs/'+label+'_20260911',regional_weight=calibration['common_weight'])
        save(label+'_manifest.json',manifest)
    save('sources.json',sources)
    print(json.dumps(sources,indent=2))

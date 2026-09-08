"""Validate and stage an already-completed, explicitly authorized Val pilot."""
import argparse
import gc
import json
import os
import subprocess
from pathlib import Path
import hf_xet
import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--source',required=True)
    args=p.parse_args()
    assert args.run.parent==Path('/root/recipe_runs')
    runtime=json.loads((args.run/'runtime.json').read_text())
    result=json.loads((args.run/'validation.json').read_text())
    assert runtime['phase']=='complete' and runtime['training_rc']==runtime['validation_rc']==0
    assert result['split']=='validation' and result['samples']==1429 and not result['test_split_accessed']
    assert result['checkpoint_git_commit']==result['analysis_git_commit']==runtime['source_git_commit']==args.source
    assert len(args.source)==40 and result['epochs']==80 and result['threshold']==.5
    repo=Path(runtime['repository'])
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.source
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
    best=Path(runtime['best_checkpoint'])
    last=best.with_name('last_model-BetterLViT.pth.tar')
    checks=[]
    for model in (best,last):
        checkpoint=torch.load(model,map_location='cpu',weights_only=False)
        assert checkpoint['source_git_commit']==args.source and checkpoint['state_dict']
        assert checkpoint['training_recipe']==result['training_recipe']
        assert checkpoint['experiment_name']==result['experiment']
        assert checkpoint['best_epoch']==result['checkpoint_best_epoch']
        assert len(checkpoint['epoch_history'])==(80 if model==last else result['checkpoint_best_epoch'])
        checks.append(dict(name=model.name,source_git_commit=args.source,best_epoch=checkpoint['best_epoch'],history_rows=len(checkpoint['epoch_history'])))
        del checkpoint
        gc.collect()
    session=best.parent.parent
    logs=list(session.glob('*.log'))
    events=list((session/'tensorboard_logs').glob('events.out*'))
    assert len(logs)==len(events)==1
    assert 'Traceback (most recent call last)' not in (args.run/'training.log').read_text()
    mapping={logs[0].name:logs[0],result['paper_id'].lower()+'_validation_evaluation.json':args.run/'validation.json',
        'models/'+best.name:best,'models/'+last.name:last,'tensorboard_logs/'+events[0].name:events[0]}
    stage=Path('/root/recipe_runs/hf_staging')/args.source[:8]
    for name,src in mapping.items():
        dst=stage/name
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():assert os.path.samefile(src,dst)
        else:os.link(src,dst)
    hashes=hf_xet.hash_files([str(src) for src in mapping.values()])
    proof=dict(source_git_commit=args.source,bucket_prefix=args.source[:8],
        classification='completed_validation_only_80e_pilot',test_split_accessed=False,
        checkpoint_checks=checks,training_and_validation_returncodes=[0,0],
        files=[dict(source=str(src),remote=args.source[:8]+'/'+name,bytes=info.file_size,xet_hash=info.hash)
            for (name,src),info in zip(mapping.items(),hashes)])
    (stage/'../').resolve().joinpath(args.source[:8]+'_manifest.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof))


if __name__=='__main__':main()

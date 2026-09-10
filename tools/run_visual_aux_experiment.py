"""Immutable S1/S2 runner: full training, scheduled telemetry, then Val export."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from visual_aux_protocol import validate_visual_aux_manifest


def write(path,value):
    temp=Path(str(path)+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1]
    manifest=json.loads((repo/'experiment_manifests/active_visual_aux.json').read_text())
    validate_visual_aux_manifest(manifest)
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip()
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    assert shutil.disk_usage(repo).free>4_000_000_000
    args.run.mkdir(parents=True,exist_ok=False)
    lock=(args.run.parent/'visual_aux_training.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert subprocess.run(['pgrep','-f','[t]rain_model.py'],capture_output=True).returncode==1
    cache='/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface'
    env=dict(os.environ,BETTERLVIT_EXPERIMENT=manifest['profile'],BETTERLVIT_SEED=str(manifest['seed']),
        BETTERLVIT_EPOCHS='80',BETTERLVIT_BATCH_SIZE='16',BETTERLVIT_TRAIN_DROP_LAST='1',
        BETTERLVIT_NUM_WORKERS='4',BETTERLVIT_DETERMINISTIC='1',BETTERLVIT_CUDNN_ENABLED='1',
        BETTERLVIT_VIS_FREQUENCY='100000',BETTERLVIT_GIT_COMMIT=sha,BETTERLVIT_RESUME_PATH='',
        BETTERLVIT_EPOCH_TIMING_PATH=str(args.run/'epoch_timing.jsonl'),
        BETTERLVIT_AUX_AUDIT_PATH=str(args.run/'diagnostics'),
        CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED=str(manifest['seed']),
        TOKENIZERS_PARALLELISM='false',TEST_SPLIT_ALLOWED='0',AUTO_TEST_EVALUATE='0',
        HF_HOME=cache,HF_HUB_CACHE=cache+'/hub',HF_MODULES_CACHE=cache+'/modules',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    python='/root/autodl-tmp/envs/betterlvit-paper/bin/python'
    state=dict(source_git_commit=sha,repository=str(repo),run=str(args.run),manifest=manifest,
        started_unix=time.time(),phase='training',test_split_accessed=False)
    write(args.run/'runtime.json',state)
    try:
        with (args.run/'training.log').open('w') as log:
            rc=subprocess.run([python,'-u','train_model.py'],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        state.update(training_rc=rc,training_ended_unix=time.time())
        write(args.run/'runtime.json',state)
        if rc:raise RuntimeError('Training failed: '+str(rc))
        bests=list((repo/'Covid19/BetterLViT'/manifest['profile']).glob('*/models/best_model-BetterLViT.pth.tar'))
        assert len(bests)==1
        assert len(list((args.run/'diagnostics').glob('epoch_*.json')))==4
        state.update(phase='validation',best_checkpoint=str(bests[0]))
        write(args.run/'runtime.json',state)
        with (args.run/'validation.log').open('w') as log:
            rc=subprocess.run([python,'tools/export_validation_metrics.py','--experiment',manifest['profile'],
                '--checkpoint',str(bests[0]),'--output',str(args.run/'validation.json'),
                '--batch-size','16','--threshold','0.5'],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        state.update(validation_rc=rc,validation_ended_unix=time.time())
        if rc:raise RuntimeError('Validation export failed: '+str(rc))
        state.update(phase='complete',completed_unix=time.time())
    except Exception as exc:
        state.update(phase='failed',error=str(exc),failed_unix=time.time())
        raise
    finally:write(args.run/'runtime.json',state)


if __name__=='__main__':main()

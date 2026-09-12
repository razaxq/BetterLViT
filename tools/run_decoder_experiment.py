"""Detached-compatible, immutable 80-epoch runner with automatic validation export."""
import argparse
import fcntl
import json
import os
import subprocess
import time
import sys
import shutil
import torch
from pathlib import Path


sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training_recipe import validate_recipe_manifest, planned_rates, rates_equal


def write(path, value):
    temp = Path(str(path) + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    manifest_path = repo / 'experiment_manifests' / 'active_decoder.json'
    manifest = json.loads(manifest_path.read_text())
    validate_recipe_manifest(manifest)
    assert manifest['profile'] in ('t1_decoder_visual','t2_decoder_text')
    assert manifest['seed']==1219 and manifest['adapter_parameters']==16896
    assert manifest['decoder_context_mode']=={'t1_decoder_visual':'visual','t2_decoder_text':'text'}[manifest['profile']]
    assert not list((repo/'Covid19').glob('**/*.pth.tar')), 'Never retrain a used worktree'
    assert shutil.disk_usage(repo).free>4_000_000_000
    assert int(subprocess.check_output(['du','-sb','/autodl-fs/data'],text=True).split()[0])<20_000_000_000
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'], cwd=repo, text=True).strip()
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip()
    args.run.mkdir(parents=True, exist_ok=False)
    lock = (args.run.parent / 'decoder_training.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if subprocess.run(['pgrep','-f','[t]rain_model.py'], capture_output=True).returncode == 0:
        raise RuntimeError('Another full segmentation training is active')
    env = dict(os.environ)
    env.update(BETTERLVIT_EXPERIMENT=manifest['profile'], BETTERLVIT_SEED=str(manifest['seed']),
        BETTERLVIT_EPOCHS='80', BETTERLVIT_BATCH_SIZE='16', BETTERLVIT_TRAIN_DROP_LAST='1',
        BETTERLVIT_NUM_WORKERS='4', BETTERLVIT_DETERMINISTIC='1', BETTERLVIT_CUDNN_ENABLED='1',
        BETTERLVIT_VIS_FREQUENCY='100000', BETTERLVIT_GIT_COMMIT=commit,
        BETTERLVIT_RESUME_PATH='',
        BETTERLVIT_EPOCH_TIMING_PATH=str(args.run/'epoch_timing.jsonl'),
        BETTERLVIT_DECODER_OBSERVATIONS_PATH=str(args.run/'decoder_observations.jsonl'),
        CUBLAS_WORKSPACE_CONFIG=':4096:8',
        PYTHONHASHSEED=str(manifest['seed']), TOKENIZERS_PARALLELISM='false',
        TEST_SPLIT_ALLOWED='0', AUTO_TEST_EVALUATE='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_HOME='/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface',
        HF_HUB_CACHE='/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface/hub',
        HF_MODULES_CACHE='/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface/modules')
    python = '/root/autodl-tmp/envs/betterlvit-paper/bin/python'
    state = {'source_git_commit':commit, 'repository':str(repo), 'run':str(args.run),
        'manifest':manifest,
        'started_unix':time.time(), 'phase':'training', 'test_split_accessed':False}
    write(args.run/'runtime.json', state)
    try:
        with (args.run/'training.log').open('w') as log:
            rc = subprocess.run([python, '-u', 'train_model.py'], cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        state.update(training_rc=rc, training_ended_unix=time.time())
        write(args.run/'runtime.json', state)
        if rc:
            raise RuntimeError('Training failed: ' + str(rc))
        bests = list((repo/'Covid19'/'BetterLViT'/manifest['profile']).glob('*/models/best_model-BetterLViT.pth.tar'))
        if len(bests) != 1:
            raise RuntimeError('Expected exactly one Best checkpoint in this frozen worktree')
        metadata=[]
        for path in (bests[0],bests[0].with_name('last_model-BetterLViT.pth.tar')):
            ck=torch.load(path,map_location='cpu',weights_only=True)
            assert ck['source_git_commit']==commit and ck['seed']==manifest['seed'] and ck['epochs']==80
            assert ck['decoder_context']['mode']==manifest['decoder_context_mode']
            if path.name.startswith('last_'):
                assert ck['epoch']==79 and len(ck['epoch_history'])==80
                assert rates_equal([h['lr'] for h in ck['epoch_history']],planned_rates('single_cosine'))
            metadata.append(dict(path=str(path),bytes=path.stat().st_size,source_git_commit=commit,
                seed=ck['seed'],epoch=ck['epoch'],epochs=ck['epochs'],best_epoch=ck['best_epoch'],
                history_rows=len(ck['epoch_history']),decoder_context=ck['decoder_context']))
            del ck
        write(args.run/'checkpoint_metadata.json',metadata)
        state.update(phase='validation', best_checkpoint=str(bests[0]))
        write(args.run/'runtime.json', state)
        with (args.run/'validation.log').open('w') as log:
            rc = subprocess.run([python, 'tools/export_validation_metrics.py', '--experiment',manifest['profile'],
                '--checkpoint',str(bests[0]),'--output',str(args.run/'validation.json'), '--batch-size','16','--threshold','0.5'],
                cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        state.update(validation_rc=rc, validation_ended_unix=time.time())
        if rc:
            raise RuntimeError('Validation export failed: ' + str(rc))
        state.update(phase='complete', completed_unix=time.time())
    except Exception as exc:
        state.update(phase='failed', error=str(exc), failed_unix=time.time())
        raise
    finally:
        write(args.run/'runtime.json',state)


if __name__ == '__main__':
    main()

"""Resume frozen P11 epoch 80 to 150, then export validation and authorized Test."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PARENT = '2fc6ab5c8e4662d741fd8b994e55b780391948ac'


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    previous = json.loads((args.parent_run / 'p11_test.json').read_text())
    parent_repo = Path(json.loads((args.parent_run / 'runtime.json').read_text())['repository'])
    parent_best = Path(previous['checkpoint'])
    resume = parent_best.with_name('last_model-BetterLViT.pth.tar')
    assert previous['checkpoint_git_commit'] == PARENT
    assert not output.exists()
    sessions = repo / 'Covid19/BetterLViT/p11_dual_grain'
    assert not list(sessions.glob('*'))
    assert shutil.disk_usage(repo).free >= 5 * 2**30
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'])
    lock = open('/root/betterlvit_race_pe.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import torch
    checkpoint = torch.load(resume, map_location='cpu', weights_only=True)
    assert checkpoint['source_git_commit'] == PARENT
    assert checkpoint['epoch'] == 79 and checkpoint['best_epoch'] == 80
    assert len(checkpoint['epoch_history']) == 80 and checkpoint['selection_metric'] == 'iou'
    assert checkpoint['dual_grain_enabled'] and checkpoint['batch_size'] == 16
    assert checkpoint['seed'] == 1219 and checkpoint['rng_state'] and checkpoint['optimizer']
    assert checkpoint['lr_scheduler']['last_epoch'] == 80
    del checkpoint
    output.mkdir(parents=True)
    env = os.environ.copy()
    env.update(BETTERLVIT_EXPERIMENT='p11_dual_grain', BETTERLVIT_SEED='1219',
        BETTERLVIT_EPOCHS='150', BETTERLVIT_BATCH_SIZE='16', BETTERLVIT_TRAIN_DROP_LAST='1',
        BETTERLVIT_NUM_WORKERS='4', BETTERLVIT_DETERMINISTIC='1', BETTERLVIT_CUDNN_ENABLED='1',
        BETTERLVIT_VIS_FREQUENCY='100000', BETTERLVIT_GIT_COMMIT=commit,
        BETTERLVIT_RESUME_PATH=str(resume), CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONHASHSEED='1219',
        HF_HOME='/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        TEST_SPLIT_ALLOWED='0', AUTO_TEST_EVALUATE='0', AUTO_EVALUATE='0', PYTHONUNBUFFERED='1')
    metadata = {'source_git_commit': commit, 'parent_source_git_commit': PARENT,
        'resume_checkpoint': str(resume), 'resume_checkpoint_sha256': sha256(resume),
        'parent_best_checkpoint': str(parent_best), 'parent_best_sha256': sha256(parent_best),
        'classification': 'user_requested_post_test_80_to_150_continuation',
        'total_epochs': 150, 'additional_epochs': 70, 'training_from_scratch': False,
        'seed': 1219, 'batch_size': 16, 'threshold': 0.5, 'selection_metric': 'validation_macro_iou',
        'scheduler_policy': 'restore unchanged T_0=10 T_mult=1 eta_min=0.0001',
        'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0),
        'versions': {n: importlib.metadata.version(n) for n in ('transformers', 'timm', 'numpy', 'peft')},
        'started_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
        'repository': str(repo), 'parent_run': str(args.parent_run),
        'auto_test_after_successful_training': True}
    (output / 'runtime.json').write_text(json.dumps(metadata, indent=2))

    def run(stage, script, *arguments, script_repo=repo):
        (output / 'chain.status').write_text(stage + '\n')
        with (output / (stage + '.log')).open('w') as log:
            process = subprocess.run([sys.executable, str(script_repo / script), *map(str, arguments)],
                                     cwd=script_repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        (output / (stage + '.status')).write_text(str(process.returncode) + '\n')
        process.check_returncode()

    try:
        run('training', 'train_model.py')
        lasts = list(sessions.glob('*/models/last_model-BetterLViT.pth.tar'))
        assert len(lasts) == 1
        last = torch.load(lasts[0], map_location='cpu', weights_only=True)
        assert last['epoch'] == 149 and len(last['epoch_history']) == 150
        assert last['source_git_commit'] == commit and last['dual_grain_enabled']
        best_epoch = last['best_epoch']
        del last
        best = lasts[0].with_name('best_model-BetterLViT.pth.tar')
        eval_repo = repo
        if best_epoch == 80:
            # No improvement after resuming: evaluate the untouched parent Best
            # with its exact original source, retaining honest provenance.
            assert not best.exists()
            best, eval_repo = parent_best, parent_repo
        else:
            assert best_epoch > 80 and best.is_file()
        (output / 'selection.json').write_text(json.dumps({
            'completed_epochs': 150, 'best_epoch': best_epoch, 'checkpoint': str(best),
            'checkpoint_sha256': sha256(best), 'evaluation_repository': str(eval_repo),
            'selection_scope': 'all 150 validation epochs', 'threshold': 0.5}, indent=2))
        run('validation', 'tools/export_validation_metrics.py', '--experiment', 'p11_dual_grain',
            '--checkpoint', best, '--output', output / 'p11_validation.json',
            '--batch-size', 16, '--threshold', .5, script_repo=eval_repo)
        env['TEST_SPLIT_ALLOWED'] = '1'
        run('test', 'tools/evaluate_fixed_test.py', '--experiment', 'p11_dual_grain',
            '--checkpoint', best, '--output', output / 'p11_test.json',
            '--batch-size', 16, '--threshold', .5, script_repo=eval_repo)
        for split in ('validation', 'test'):
            run('compare_' + split, 'tools/compare_p11_continuation.py',
                '--parent', args.parent_run / ('p11_' + split + '.json'),
                '--candidate', output / ('p11_' + split + '.json'),
                '--selection', output / 'selection.json',
                '--output', output / ('p11_80_vs_150_' + split + '.json'))
        (output / 'chain.status').write_text('complete\n')
    except BaseException:
        (output / 'chain.status').write_text('failed\n')
        raise


if __name__ == '__main__':
    main()

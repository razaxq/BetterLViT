"""Frozen P11 training -> validation export -> authorized fixed-threshold Test."""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--control-validation', type=Path, required=True)
    parser.add_argument('--control-test', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    manifest = json.loads((repo / 'experiment_manifests/p11_dual_grain.json').read_text())
    assert args.control_validation.is_file() and args.control_test.is_file()
    assert not output.exists()
    assert not list((repo / 'Covid19/BetterLViT/p11_dual_grain').glob('*'))
    assert shutil.disk_usage(repo).free >= 5 * 2**30
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'])
    lock = open('/root/betterlvit_race_pe.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output.mkdir(parents=True)
    env = os.environ.copy()
    env.update(BETTERLVIT_EXPERIMENT=manifest['profile'], BETTERLVIT_SEED='1219',
        BETTERLVIT_EPOCHS='80', BETTERLVIT_BATCH_SIZE='16', BETTERLVIT_TRAIN_DROP_LAST='1',
        BETTERLVIT_NUM_WORKERS='4', BETTERLVIT_DETERMINISTIC='1', BETTERLVIT_CUDNN_ENABLED='1',
        BETTERLVIT_VIS_FREQUENCY='100000', BETTERLVIT_GIT_COMMIT=commit,
        BETTERLVIT_RESUME_PATH='', CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONHASHSEED='1219',
        HF_HOME='/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        TEST_SPLIT_ALLOWED='0', AUTO_TEST_EVALUATE='0', AUTO_EVALUATE='0', PYTHONUNBUFFERED='1')
    import torch
    metadata = {'source_git_commit': commit, 'manifest': manifest, 'python': sys.version,
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
        'versions': {name: importlib.metadata.version(name) for name in ('transformers', 'timm', 'numpy', 'peft')},
        'started_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
        'repository': str(repo), 'auto_test_after_successful_training': True,
        'control_validation': str(args.control_validation.resolve()), 'control_test': str(args.control_test.resolve())}
    (output / 'runtime.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    def run(stage, script, *arguments):
        (output / 'chain.status').write_text(stage + '\n')
        with (output / (stage + '.log')).open('w') as log:
            process = subprocess.run([sys.executable, str(repo / script), *map(str, arguments)],
                                     cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        (output / (stage + '.status')).write_text(str(process.returncode) + '\n')
        process.check_returncode()
    try:
        run('training', 'train_model.py')
        checkpoints = list((repo / 'Covid19/BetterLViT/p11_dual_grain').glob('*/models/best_model-BetterLViT.pth.tar'))
        assert len(checkpoints) == 1
        checkpoint = checkpoints[0]
        last = torch.load(checkpoint.with_name('last_model-BetterLViT.pth.tar'), map_location='cpu', weights_only=True)
        assert last['epoch'] == 79 and last['source_git_commit'] == commit and last['dual_grain_enabled']
        del last
        run('validation', 'tools/export_validation_metrics.py', '--experiment', manifest['profile'],
            '--checkpoint', checkpoint, '--output', output / 'p11_validation.json', '--batch-size', 16, '--threshold', .5)
        # This new experiment has explicit user authorization for automatic Test.
        env['TEST_SPLIT_ALLOWED'] = '1'
        run('test', 'tools/evaluate_fixed_test.py', '--experiment', manifest['profile'],
            '--checkpoint', checkpoint, '--output', output / 'p11_test.json', '--batch-size', 16, '--threshold', .5)
        run('compare_validation', 'tools/compare_dual_grain.py', '--control', args.control_validation,
            '--candidate', output / 'p11_validation.json', '--output', output / 'c4_vs_p11_validation.json')
        run('compare_test', 'tools/compare_dual_grain.py', '--control', args.control_test,
            '--candidate', output / 'p11_test.json', '--output', output / 'c4_vs_p11_test.json')
        (output / 'chain.status').write_text('complete\n')
    except BaseException:
        (output / 'chain.status').write_text('failed\n')
        raise


if __name__ == '__main__':
    main()

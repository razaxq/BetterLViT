"""Evaluate frozen S1/S2 Best checkpoints sequentially, without any training."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def run(directory, analysis_sha):
    directory = Path(directory).resolve()
    plan = json.loads((directory / 'authorization.json').read_text())
    state = dict(phase='starting', evaluation_source_git_commit=analysis_sha,
                 started_unix=time.time(), training_started=False, arms=[])
    state_path = directory / 'runtime.json'
    write(state_path, state)
    try:
        cache = '/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface'
        for arm in plan['arms']:
            repo = arm['repository']
            assert subprocess.check_output(['git', '-C', repo, 'rev-parse', 'HEAD'], text=True).strip() == arm['source_git_commit']
            assert not subprocess.check_output(['git', '-C', repo, 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
            label = arm['label']
            current = dict(label=label, started_unix=time.time(), source_git_commit=arm['source_git_commit'])
            state['arms'].append(current)
            state.update(phase='evaluating', current_label=label)
            write(state_path, state)
            env = dict(os.environ, HF_HOME=cache, HF_HUB_CACHE=cache + '/hub', HF_MODULES_CACHE=cache + '/modules',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                       CUBLAS_WORKSPACE_CONFIG=':4096:8', BETTERLVIT_EXPERIMENT=arm['profile'],
                       BETTERLVIT_SEED=str(arm['seed']), PYTHONHASHSEED=str(arm['seed']), BETTERLVIT_EPOCHS='80',
                       BETTERLVIT_BATCH_SIZE='16', BETTERLVIT_DETERMINISTIC='1', BETTERLVIT_CUDNN_ENABLED='1',
                       BETTERLVIT_GIT_COMMIT=arm['source_git_commit'], TEST_SPLIT_ALLOWED='1', AUTO_TEST_EVALUATE='0',
                       RECIPE_EVAL_REPO=repo, RECIPE_TEST_BEST_EPOCH=str(arm['best_epoch']),
                       RECIPE_TEST_ANALYSIS_SHA=analysis_sha,
                       VISUAL_AUX_TEST_AUTHORIZATION=str(directory / 'authorization.json'),
                       VISUAL_AUX_TEST_AUTHORIZATION_SHA256=sha256(directory / 'authorization.json'))
            output = directory / (label + '_test.json')
            with (directory / (label + '_test.log')).open('x') as log:
                rc = subprocess.run([sys.executable, '-u', str(directory / 'evaluate_test.py'),
                                     '--experiment', arm['profile'], '--checkpoint', arm['checkpoint'],
                                     '--output', str(output), '--batch-size', '16', '--threshold', '0.5'],
                                    cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900).returncode
            current.update(returncode=rc, ended_unix=time.time())
            write(state_path, state)
            assert rc == 0, f'{label} failed; see its log'
            result = json.loads(output.read_text())
            assert result['samples'] == 2113 and result['split'] == 'test' and result['threshold'] == .5
            assert result['checkpoint_git_commit'] == result['analysis_git_commit'] == arm['source_git_commit']
            assert result['evaluation_source_git_commit'] == analysis_sha
            assert result['checkpoint_best_epoch'] == arm['best_epoch'] and result['seed'] == 1219
            assert result['evaluation_script_sha256'] == sha256(directory / 'evaluate_test.py')
            assert result['authorization_sha256'] == sha256(directory / 'authorization.json')
            assert not subprocess.check_output(['git', '-C', repo, 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
            current['result_sha256'] = sha256(output)
        state.update(phase='complete', completed_unix=time.time(), current_label=None)
        state['artifacts'] = {p.name: dict(bytes=p.stat().st_size, sha256=sha256(p))
                              for p in directory.glob('*_test.*') if p.suffix in ('.json', '.log')}
        write(state_path, state)
    except BaseException:
        state.update(phase='failed', failed_unix=time.time(), error=traceback.format_exc())
        write(state_path, state)
        raise


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2])

"""Detached sequential inference of the six frozen, validation-selected Best models."""
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from protocol import load_plan, sha256, validate_result, write_json


def run(directory, analysis_sha):
    directory = Path(directory).resolve()
    assert len(analysis_sha) == 40 and all(c in '0123456789abcdef' for c in analysis_sha)
    plan = load_plan(directory)
    lock = Path('/root/recipe_runs/recipe_training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = directory / 'runtime.json'
    assert not state_path.exists(), 'Never overwrite an existing Test execution'
    state = dict(phase='preflight', started_unix=time.time(), evaluation_source_git_commit=analysis_sha,
                 test_plan_sha256=sha256(directory / 'test_plan.json'), arms=[], test_split_accessed=False)
    write_json(state_path, state)
    try:
        for arm in plan['arms']:
            repo = arm['repository']
            assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() == arm['source_git_commit']
            assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=repo, text=True).strip()
            assert Path(arm['checkpoint']).is_file()
            assert not (directory / (arm['label'] + '_test.json')).exists()
        cache = '/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface'
        for arm in plan['arms']:
            label = arm['label']
            current = dict(label=label, started_unix=time.time(), source_git_commit=arm['source_git_commit'])
            state.update(phase='evaluating', current_label=label, test_split_accessed=True)
            state['arms'].append(current)
            write_json(state_path, state)
            env = dict(os.environ, HF_HOME=cache, HF_HUB_CACHE=cache + '/hub', HF_MODULES_CACHE=cache + '/modules',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                       CUBLAS_WORKSPACE_CONFIG=':4096:8', BETTERLVIT_EXPERIMENT=arm['profile'],
                       BETTERLVIT_SEED=str(arm['seed']), PYTHONHASHSEED=str(arm['seed']), BETTERLVIT_EPOCHS='80',
                       BETTERLVIT_BATCH_SIZE='16', BETTERLVIT_DETERMINISTIC='1', BETTERLVIT_CUDNN_ENABLED='1',
                       BETTERLVIT_GIT_COMMIT=arm['source_git_commit'], TEST_SPLIT_ALLOWED='1', AUTO_TEST_EVALUATE='0',
                       RECIPE_EVAL_REPO=arm['repository'], RECIPE_TEST_GATE=str(directory / 'three_seed_summary.json'),
                       RECIPE_TEST_GATE_SHA256=plan['gate_sha256'], RECIPE_TEST_BEST_EPOCH=str(arm['best_epoch']),
                       RECIPE_TEST_ANALYSIS_SHA=analysis_sha)
            output = directory / (label + '_test.json')
            with (directory / (label + '_test.log')).open('x') as log:
                rc = subprocess.run([sys.executable, '-u', str(directory / 'evaluate_test.py'),
                                     '--experiment', arm['profile'], '--checkpoint', arm['checkpoint'],
                                     '--output', str(output), '--batch-size', '16', '--threshold', '0.5'],
                                    cwd=arm['repository'], env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            current.update(returncode=rc, ended_unix=time.time())
            write_json(state_path, state)
            assert rc == 0, f'{label} evaluation failed; see its log'
            validate_result(json.loads(output.read_text()), arm, plan, analysis_sha, sha256(directory / 'evaluate_test.py'))
            current['result_sha256'] = sha256(output)
        state['phase'] = 'summarizing'
        write_json(state_path, state)
        from compare_test import summarize
        write_json(directory / 'three_seed_test_summary.json', summarize(directory, analysis_sha))
        state.update(phase='complete', completed_unix=time.time(), current_label=None)
        state['artifacts'] = {p.name: dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in directory.iterdir()
                              if p.name.endswith(('_test.json', '_test.log')) or p.name == 'three_seed_test_summary.json'}
        write_json(state_path, state)
    except BaseException:
        state.update(phase='failed', failed_unix=time.time(), error=traceback.format_exc())
        write_json(state_path, state)
        raise


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2])

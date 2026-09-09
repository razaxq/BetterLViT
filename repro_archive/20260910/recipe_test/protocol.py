"""Frozen six-checkpoint Test protocol and result integrity checks."""
import hashlib
import json
import math
import statistics
from pathlib import Path, PurePosixPath

SEEDS = [1219, 2027, 3407]
METRICS = ('iou', 'dice', 'precision', 'recall', 'brier')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    temporary.replace(path)


def load_plan(directory):
    directory = Path(directory)
    plan = json.loads((directory / 'test_plan.json').read_text(encoding='utf-8'))
    gate_path = directory / 'three_seed_summary.json'
    gate = json.loads(gate_path.read_text(encoding='utf-8'))
    assert sha256(gate_path) == plan['gate_sha256'], 'Gate hash mismatch'
    assert gate['passed'] and all(gate['checks'].values()), 'Validation gate failed'
    assert gate['predeclared_seeds'] == SEEDS
    assert plan['samples'] == 2113 and plan['batch_size'] == 16
    assert plan['threshold'] == 0.5 and plan['epochs'] == 80
    assert plan['selection_metric'] == 'iou'
    assert [a['label'] for a in plan['arms']] == [f'{r}s{s}' for s in SEEDS for r in ('c4', 'r2')]
    for arm in plan['arms']:
        pair = next(p for p in gate['paired_runs'] if p['seed'] == arm['seed'])
        role = arm['role']
        assert arm['source_git_commit'] == pair['control_commit' if role == 'c4' else 'candidate_commit']
        assert arm['profile'] == ('c4_race_pe_control' if role == 'c4' else 'r2_single_cosine')
        assert 1 <= arm['best_epoch'] <= 80
        assert PurePosixPath(arm['checkpoint']).is_relative_to(PurePosixPath(arm['repository']))
        assert PurePosixPath(arm['checkpoint']).name == 'best_model-BetterLViT.pth.tar'
    return plan


def validate_result(result, arm, plan, analysis_sha, evaluator_sha):
    assert result['split'] == 'test' and result['test_split_accessed']
    assert result['user_authorized_test'] and not result['threshold_selected_on_test']
    for key, expected in {'seed': arm['seed'], 'experiment': arm['profile'],
                          'checkpoint': arm['checkpoint'], 'checkpoint_git_commit': arm['source_git_commit'],
                          'analysis_git_commit': arm['source_git_commit'], 'checkpoint_best_epoch': arm['best_epoch'],
                          'evaluation_source_git_commit': analysis_sha, 'evaluation_script_sha256': evaluator_sha,
                          'gate_sha256': plan['gate_sha256'], 'epochs': 80, 'threshold': 0.5,
                          'selection_metric': 'iou', 'samples': 2113}.items():
        assert result[key] == expected, (key, result[key], expected)
    for key in ('race_pe_enabled', 'race_enabled', 'text_use_lora', 'bcdh_enabled', 'cdrr_enabled', 'boundary_loss_weight'):
        assert not result[key], key
    assert len(result['checkpoint_sha256']) == 64
    records = result['records']
    assert len(records) == 2113 and len({r['name'] for r in records}) == 2113
    for metric in METRICS:
        values = [r[metric] for r in records]
        assert all(math.isfinite(v) and 0 <= v <= 1 for v in values)
        assert math.isclose(statistics.mean(values), result['macro_' + metric], abs_tol=1e-12)
    return {r['name']: r for r in records}

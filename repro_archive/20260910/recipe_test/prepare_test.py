"""Freeze Test model paths solely from completed validation exports."""
import json
import shutil
import statistics
from pathlib import Path
from protocol import load_plan, sha256, write_json

HERE = Path(__file__).resolve().parent
ROOT = Path('D:/BetterLViT')
REPLICATION = HERE.parents[1] / '20260909/recipe_replication'


def main():
    assert not (HERE / 'test_plan.json').exists(), 'Test plan is immutable once prepared'
    shutil.copyfile(ROOT / 'outputs/recipe_replication_20260909/three_seed_summary.json', HERE / 'three_seed_summary.json')
    sources = json.loads((REPLICATION / 'sources.json').read_text())
    arms, durations = [], []
    for seed in (1219, 2027, 3407):
        for role in ('c4', 'r2'):
            label = f'{role}s{seed}'
            if seed == 1219:
                validation = (ROOT / 'experiment_docs_work/repro_archive/20260908/visual_prior_execution/c4_validation.json'
                              if role == 'c4' else ROOT / 'outputs/recipe_20260908/r2_validation.json')
                repo = '/root/BetterLViT-race-pe-c4' if role == 'c4' else '/root/BetterLViT-recipe-r2'
                tag = ('pilot-c4-race-pe-control-80e-seed1219-20260906' if role == 'c4'
                       else 'experiment-r2-recipe-80e-seed1219-20260908-v2')
            else:
                validation = ROOT / f'outputs/recipe_replication_20260909/{label}_validation.json'
                repo, tag = sources[label]['repository'], sources[label]['experiment_tag']
                runtime = json.loads((REPLICATION / f'{label}_results/runtime.json').read_text())
                assert runtime['phase'] == 'complete' and runtime['training_rc'] == runtime['validation_rc'] == 0
                durations.append(dict(label=label, seconds=runtime['validation_ended_unix'] - runtime['training_ended_unix']))
            result = json.loads(validation.read_text(encoding='utf-8'))
            assert result['samples'] == 1429 and result['seed'] == seed and result['epochs'] == 80
            assert result['selection_metric'] == 'iou' and result['threshold'] == 0.5
            arms.append(dict(label=label, role=role, seed=seed, repository=repo, profile=result['experiment'],
                             source_git_commit=result['checkpoint_git_commit'], experiment_tag=tag,
                             checkpoint=result['checkpoint'], best_epoch=result['checkpoint_best_epoch'],
                             validation_source=str(validation), validation_sha256=sha256(validation)))
    estimate = statistics.mean(d['seconds'] for d in durations) * 2113 / 1429 + 20
    plan = dict(version=1, samples=2113, batch_size=16, threshold=0.5, epochs=80, selection_metric='iou',
                gate_sha256=sha256(HERE / 'three_seed_summary.json'), arms=arms,
                forecast=dict(measured_validation_seconds=durations, validation_samples=1429,
                              estimated_seconds_per_test_model=estimate, estimated_chain_seconds=6 * estimate,
                              completion_check_buffer_seconds=720),
                authorization='User-approved training recipe plan and conditional Test phase after 3 paired seeds pass.',
                historical_test_access=True, threshold_selected_on_test=False,
                c4_1219_re_evaluated_for_uniform_export=True)
    write_json(HERE / 'test_plan.json', plan)
    load_plan(HERE)
    print(json.dumps(dict(arms=len(arms), estimated_chain_seconds=6 * estimate, gate_passed=True)))


if __name__ == '__main__':
    main()

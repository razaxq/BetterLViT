"""Report all fixed paired Test results; never select a new model or threshold."""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from protocol import METRICS, SEEDS, load_plan, sha256, validate_result, write_json


def summarize(directory, analysis_sha):
    directory = Path(directory)
    plan = load_plan(directory)
    loaded, records = {}, {}
    for arm in plan['arms']:
        label = arm['label']
        result = json.loads((directory / (label + '_test.json')).read_text(encoding='utf-8'))
        records[label] = validate_result(result, arm, plan, analysis_sha, sha256(directory / 'evaluate_test.py'))
        loaded[label] = result
    reference = records['c4s1219']
    for current in records.values():
        assert current.keys() == reference.keys(), 'Test image IDs mismatch'
        assert all(current[n]['label_pixels'] == reference[n]['label_pixels'] for n in reference), 'Test labels mismatch'
    pairs = []
    names = sorted(reference)
    for seed in SEEDS:
        c4, r2 = loaded[f'c4s{seed}'], loaded[f'r2s{seed}']
        delta = {m: r2['macro_' + m] - c4['macro_' + m] for m in METRICS}
        ci = {}
        rng = np.random.default_rng(seed)
        arrays = np.asarray([[records[f'r2s{seed}'][n][m] - records[f'c4s{seed}'][n][m] for n in names]
                             for m in ('iou', 'dice')])
        estimates = []
        for _ in range(100):
            index = rng.integers(0, len(names), size=(100, len(names)))
            estimates.append(arrays[:, index].mean(axis=-1))
        distribution = np.concatenate(estimates, axis=1)
        for i, metric in enumerate(('iou', 'dice')):
            ci[metric] = np.quantile(distribution[i], [0.025, 0.975]).tolist()
        pairs.append(dict(seed=seed, control_commit=c4['checkpoint_git_commit'], candidate_commit=r2['checkpoint_git_commit'],
                          control={m: c4['macro_' + m] for m in METRICS},
                          candidate={m: r2['macro_' + m] for m in METRICS}, deltas=delta,
                          paired_image_bootstrap_95ci=ci))
    result = dict(split='test', test_split_accessed=True, samples_per_model=2113, seeds=SEEDS,
                  evaluation_source_git_commit=analysis_sha, threshold=0.5, selection_metric='validation_macro_iou',
                  paired_runs=pairs, delta_summary={}, arm_summary={},
                  all_three_iou_deltas_positive=all(p['deltas']['iou'] > 0 for p in pairs),
                  interpretation='All three registered seeds reported. Image-bootstrap intervals are conditional on fixed models, not cross-seed confidence intervals. This Test split has been accessed in historical development; it is not a new independent holdout. LR recipe optimization is not a second structural innovation.')
    for metric in METRICS:
        values = [p['deltas'][metric] for p in pairs]
        result['delta_summary'][metric] = dict(mean=statistics.mean(values), sample_std=statistics.stdev(values))
        result['arm_summary'][metric] = {}
        for role in ('control', 'candidate'):
            values = [p[role][metric] for p in pairs]
            result['arm_summary'][metric][role] = dict(mean=statistics.mean(values), sample_std=statistics.stdev(values))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--analysis-sha', required=True)
    args = parser.parse_args()
    result = summarize(args.directory, args.analysis_sha)
    write_json(args.directory / 'three_seed_test_summary.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'paired_runs'}))

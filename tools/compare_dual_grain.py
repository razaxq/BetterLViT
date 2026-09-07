"""Paired macro-IoU comparison of fixed-threshold, matching-protocol exports."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    control, candidate = [json.loads(path.read_text()) for path in (args.control, args.candidate)]
    for key in ('split', 'samples', 'seed', 'epochs', 'threshold', 'selection_metric'):
        assert control[key] == candidate[key], key
    assert control['threshold'] == 0.5
    assert control['checkpoint_git_commit'] == 'add4908a0d6f702b0a10c4581725b535543829b8'
    assert len(candidate['checkpoint_git_commit']) == 40
    assert candidate['experiment'] == 'p11_dual_grain'
    left, right = [{x['name']: x for x in p['records']} for p in (control, candidate)]
    assert set(left) == set(right) and len(left) == control['samples']
    names = sorted(left)
    assert all(left[n]['label_pixels'] == right[n]['label_pixels'] for n in names)
    metrics = {}
    for metric in ('iou', 'dice', 'precision', 'recall'):
        a = np.asarray([left[n][metric] for n in names])
        b = np.asarray([right[n][metric] for n in names])
        delta = b - a
        assert np.isfinite(delta).all()
        rng = np.random.default_rng(1219)
        means = np.asarray([delta[rng.integers(len(delta), size=len(delta))].mean() for _ in range(10000)])
        metrics[metric] = {'control': float(a.mean()), 'candidate': float(b.mean()),
            'delta': float(delta.mean()), 'paired_case_bootstrap_95_ci': np.quantile(means, [.025, .975]).tolist()}
    small = sorted(names, key=lambda n: (left[n]['label_pixels'], n))[:(len(names) + 3) // 4]
    result = {'split': control['split'], 'primary_metric': 'macro_iou',
        'control_commit': control['checkpoint_git_commit'], 'candidate_commit': candidate['checkpoint_git_commit'],
        'samples': len(names), 'threshold': .5, 'metrics': metrics,
        'smallest_area_quartile_iou_delta': float(np.mean([right[n]['iou'] - left[n]['iou'] for n in small])),
        'single_seed_iou_increased': metrics['iou']['delta'] > 0,
        'stable_gain_proven': False,
        'limitation': 'Case bootstrap does not establish stability across training seeds; Test is exploratory.'}
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

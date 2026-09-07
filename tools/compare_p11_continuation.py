"""Descriptive paired comparison of P11 80 versus 150 epoch continuation."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    for key in ('parent', 'candidate', 'selection', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    parent, candidate, selection = [json.loads(p.read_text()) for p in
                                    (args.parent, args.candidate, args.selection)]
    for key in ('split', 'samples', 'seed', 'threshold', 'selection_metric', 'architecture_version'):
        assert parent[key] == candidate[key], key
    assert parent['experiment'] == candidate['experiment'] == 'p11_dual_grain'
    assert parent['epochs'] == 80 and selection['completed_epochs'] == 150
    assert selection['best_epoch'] == candidate['checkpoint_best_epoch']
    assert candidate['epochs'] == (80 if selection['best_epoch'] == 80 else 150)
    assert parent['threshold'] == 0.5
    left, right = [{r['name']: r for r in p['records']} for p in (parent, candidate)]
    assert set(left) == set(right) and len(left) == parent['samples']
    names = sorted(left)
    assert all(left[n]['label_pixels'] == right[n]['label_pixels'] for n in names)
    metrics = {}
    for metric in ('iou', 'dice', 'precision', 'recall'):
        a = np.array([left[n][metric] for n in names])
        b = np.array([right[n][metric] for n in names])
        delta = b - a
        assert np.isfinite(delta).all()
        rng = np.random.default_rng(1219)
        means = [delta[rng.integers(len(delta), size=len(delta))].mean() for _ in range(10000)]
        metrics[metric] = {'parent_80': float(a.mean()), 'continued_150': float(b.mean()),
            'delta': float(delta.mean()), 'paired_case_bootstrap_95_ci': np.quantile(means, [.025, .975]).tolist()}
    result = {'split': parent['split'], 'comparison': 'P11 80 versus continued 150',
        'completed_epochs': 150, 'best_epoch': selection['best_epoch'], 'metrics': metrics,
        'parent_commit': parent['checkpoint_git_commit'],
        'candidate_checkpoint_commit': candidate['checkpoint_git_commit'],
        'stable_gain_proven': False,
        'limitation': 'Post-Test user-requested extension, single seed; unequal budgets. Not a matched C4 ablation.'}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

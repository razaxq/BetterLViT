"""CPU numerical feasibility checks, not a trained segmentation method.

Projects candidate logits onto the original prediction's SOFT foreground mass.
No dataset, model, or ground-truth-derived volume enters the projection.
Requires numpy; outputs synthetic counterexamples as well as positive examples.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def sigmoid(x):
    return np.exp(-np.logaddexp(0.0, -np.asarray(x, dtype=np.float64)))


def redistribute(z, delta):
    z = np.asarray(z, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    if z.shape != delta.shape or not np.isfinite(z).all() or not np.isfinite(delta).all():
        raise ValueError('Finite equal-shaped arrays are required')
    target = sigmoid(z).sum()
    a = z + delta
    # +/- max(abs(delta)) brackets the root by monotonicity.
    bound = float(np.max(np.abs(delta))) + 1.0
    lo, hi = -bound, bound
    for _ in range(80):
        b = (lo + hi) / 2
        if sigmoid(a + b).sum() < target:
            lo = b
        else:
            hi = b
    return sigmoid(a + (lo + hi) / 2)


def hard_iou(p, y):
    pred = np.asarray(p) >= .5
    y = np.asarray(y, dtype=bool)
    union = np.logical_or(pred, y).sum()
    return float(np.logical_and(pred, y).sum()/union) if union else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(1219)
    z = rng.normal(size=196)
    delta = rng.normal(scale=.2, size=196)
    q = redistribute(z, delta)
    w = q*(1-q)
    direction = rng.normal(size=z.shape)
    # Implicit derivative with respect to delta, holding baseline z fixed.
    analytic = w*(direction - np.dot(w, direction)/w.sum())
    eps = 1e-5
    finite = (redistribute(z, delta+eps*direction)-redistribute(z, delta-eps*direction))/(2*eps)
    ptoy = np.array([.45, .45, .55, .55])
    ztoy = np.log(ptoy/(1-ptoy))
    truth = [1, 1, 0, 0]
    useful = redistribute(ztoy, np.array([.4, .4, -.4, -.4]))
    harmful = redistribute(np.log(useful/(1-useful)), np.array([-.8, -.8, .8, .8]))
    pcount = np.full(4, .49)
    qcount = np.array([.6, .6, .38, .38])
    values = {
        'status': 'synthetic_numerical_checks_only_not_model_results',
        'mass_absolute_error': float(abs(q.sum()-sigmoid(z).sum())),
        'identity_max_error': float(abs(redistribute(z, np.zeros_like(z))-sigmoid(z)).max()),
        'uniform_delta_invariance_max_error': float(abs(redistribute(z, delta+3)-q).max()),
        'implicit_delta_jvp_finite_difference_max_error': float(abs(analytic-finite).max()),
        'toy_correct_local_evidence': {'before_iou': hard_iou(ptoy, truth), 'after_iou': hard_iou(useful, truth), 'before_soft_mass': float(ptoy.sum()), 'after_soft_mass': float(useful.sum())},
        'toy_wrong_local_evidence': {'before_iou': hard_iou(useful, truth), 'after_iou': hard_iou(harmful, truth), 'note': 'Mass preservation alone does not prevent incorrect spatial assignment.'},
        'soft_mass_does_not_preserve_hard_area': {'before_mass': float(pcount.sum()), 'after_mass': float(qcount.sum()), 'before_hard_pixels': int((pcount>=.5).sum()), 'after_hard_pixels': int((qcount>=.5).sum())},
        'not_tested': ['Learned text-image affinity', 'PyTorch/CUDA autograd implementation', 'End-to-end training', 'Real-data Dice or IoU gain'],
    }
    assert values['mass_absolute_error'] < 1e-10
    assert values['identity_max_error'] < 1e-12
    assert values['uniform_delta_invariance_max_error'] < 1e-12
    assert values['implicit_delta_jvp_finite_difference_max_error'] < 1e-7
    assert values['toy_correct_local_evidence']['after_iou'] > values['toy_correct_local_evidence']['before_iou']
    assert values['soft_mass_does_not_preserve_hard_area']['before_hard_pixels'] != values['soft_mass_does_not_preserve_hard_area']['after_hard_pixels']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(values, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(values, indent=2))


if __name__ == '__main__':
    main()

"""Paired validation gate for the frozen-visual pilot; never opens Test data."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def compare(control, candidate, role='c4'):
    for result in (control, candidate):
        if result['split'] != 'validation' or result['test_split_accessed']:
            raise ValueError('Validation-only gate rejects Test results')
        if result['checkpoint_git_commit'] != result['analysis_git_commit']:
            raise ValueError('Checkpoint/analysis provenance mismatch')
        if len(result['checkpoint_git_commit']) != 40:
            raise ValueError('Full source SHA is required')
        if result['selection_metric'] != 'iou' or result['threshold'] != .5:
            raise ValueError('Selection or prediction protocol mismatch')
        if result['epochs'] != 80 or result['samples'] != 1429:
            raise ValueError('Wrong pilot budget or sample count')
        if result['text_use_lora'] or result['boundary_loss_weight'] != 0:
            raise ValueError('Disallowed training configuration')
    if control['seed'] != candidate['seed']:
        raise ValueError('Seeds must be paired')
    if role == 'c4' and control['experiment'] != 'c4_race_pe_control':
        raise ValueError('C4 gate requires the registered C4 control')
    if role == 'random' and control['experiment'] != 'c9_visual_random':
        raise ValueError('Random-feature gate requires C9')
    a = {r['name']:r for r in control['records']}
    b = {r['name']:r for r in candidate['records']}
    if len(a) != 1429 or len(b) != 1429 or a.keys() != b.keys():
        raise ValueError('Missing, duplicate or unpaired images')
    names = sorted(a)
    areas = np.array([a[n]['label_pixels'] for n in names])
    if not np.array_equal(areas, [b[n]['label_pixels'] for n in names]):
        raise ValueError('Mask area/pairing mismatch')
    small = areas <= np.quantile(areas, .25)
    output = {'split':'validation', 'test_split_accessed':False, 'role':role,
        'control_commit':control['checkpoint_git_commit'], 'candidate_commit':candidate['checkpoint_git_commit'],
        'seed':control['seed'], 'samples':len(names), 'small_group_count':int(small.sum()),
        'small_group_area_cutoff':float(np.quantile(areas,.25)), 'deltas':{}}
    rng = np.random.default_rng(20260908)
    for key in ('iou','dice','precision','recall'):
        av = np.array([a[n][key] for n in names], dtype='float64')
        bv = np.array([b[n][key] for n in names], dtype='float64')
        if not np.isfinite(av).all() or not np.isfinite(bv).all():
            raise ValueError('Nonfinite per-image metric')
        if not np.isclose(av.mean(),control['macro_'+key], atol=1e-12, rtol=0) or not np.isclose(bv.mean(),candidate['macro_'+key], atol=1e-12, rtol=0):
            raise ValueError('Aggregate does not match per-image records')
        delta = bv-av
        boot = []
        for _ in range(20):
            boot.extend(delta[rng.integers(0,len(names),size=(500,len(names)))].mean(1))
        output['deltas'][key] = {'mean':float(delta.mean()), 'small_group_mean':float(delta[small].mean()),
            'paired_image_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist()}
    d = output['deltas']
    if role == 'c4':
        checks = {'iou_at_least_0.003':d['iou']['mean'] >= .003,
            'iou_ci_positive':d['iou']['paired_image_bootstrap_95ci'][0] > 0,
            'dice_point_not_lower':d['dice']['mean'] >= 0,
            'small_iou_point_not_lower':d['iou']['small_group_mean'] >= 0}
    else:
        checks = {'iou_beats_random_features':d['iou']['mean'] > 0}
    output.update(checks=checks, passed=all(checks.values()),
        interpretation='Single training seed; image bootstrap is not evidence of cross-seed stability')
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--role',choices=('c4','random'),default='c4')
    args = parser.parse_args()
    result = compare(json.loads(args.control.read_text()),json.loads(args.candidate.read_text()),args.role)
    result['source_json_sha256'] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.control,args.candidate)}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

"""Recompute paired validation error groups from existing exports, with no inference."""
import hashlib
import json
from pathlib import Path
from statistics import mean

DEST = Path(__file__).resolve().parent
PROJECT = Path('D:/BetterLViT')
PATHS = {
    'C4': PROJECT / 'outputs/race_pe_v2_results_20260907/c4_validation.json',
    'P11': DEST.parents[1] / '20260907/dual_grain_results/p11_validation.json',
}


def summarize(records):
    fp, fn = [], []
    for r in records:
        tp = round(r['recall'] * r['label_pixels'])
        fp.append(r['prediction_pixels'] - tp)
        fn.append(r['label_pixels'] - tp)
        assert 0 <= tp <= min(r['label_pixels'], r['prediction_pixels'])
    return {'samples': len(records), 'macro_iou': mean(r['iou'] for r in records),
            'macro_precision': mean(r['precision'] for r in records),
            'macro_recall': mean(r['recall'] for r in records),
            'mean_fp_pixels': mean(fp), 'mean_fn_pixels': mean(fn)}


def main():
    data = {name: json.loads(p.read_text(encoding='utf-8')) for name, p in PATHS.items()}
    for p in data.values():
        assert p['split'] == 'validation' and p['samples'] == 1429
        assert p['threshold'] == 0.5 and p['epochs'] == 80 and p['seed'] == 1219
    records = {key: {r['name']: r for r in p['records']} for key, p in data.items()}
    assert records['C4'].keys() == records['P11'].keys()
    names = sorted(records['C4'], key=lambda n: (records['C4'][n]['label_pixels'], n))
    assert all(records['C4'][n]['label_pixels'] == records['P11'][n]['label_pixels'] for n in names)
    groups = {}
    for index, (start, end) in enumerate([(0, 358), (358, 715), (715, 1072), (1072, 1429)], 1):
        selected = names[start:end]
        values = {key: summarize([p[n] for n in selected]) for key, p in records.items()}
        values['iou_delta_percentage_points'] = 100 * (values['P11']['macro_iou'] - values['C4']['macro_iou'])
        groups[f'area_quartile_{index}'] = values
    result = {'split': 'validation', 'new_inference_performed': False,
        'test_used_for_group_analysis': False,
        'sources': {key: {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                         'checkpoint_commit': data[key]['checkpoint_git_commit']} for key, p in PATHS.items()},
        'grouping': 'GT total lesion area per image; diagnostic only, never an inference input',
        'fp_fn_method': 'TP=round(recall*GT area); FP=predicted area-TP; FN=GT area-TP',
        'all': {key: summarize(list(p.values())) for key, p in records.items()},
        'groups': groups}
    (DEST / 'validation_error_groups.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'all': result['all'], 'groups': groups}, indent=2))


if __name__ == '__main__':
    main()

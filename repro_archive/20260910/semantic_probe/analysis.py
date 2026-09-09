"""Label-free region selection and independently reproducible diagnostic gates."""
import hashlib
import json
from pathlib import Path
import numpy as np

SELECTORS = ('disagreement', 'uncertainty', 'texture', 'random')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def blocks(a):
    a = np.asarray(a)
    assert a.shape == (224, 224)
    return a.reshape(14, 16, 14, 16).transpose(0, 2, 1, 3).reshape(196, 256)


def selected_mask(scores, k=40):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    assert scores.shape == (196,) and np.isfinite(scores).all()
    ix = np.lexsort((np.arange(196), -scores))[:k]
    low = np.zeros(196, dtype=bool)
    low[ix] = True
    mask = np.repeat(np.repeat(low.reshape(14, 14), 16, 0), 16, 1)
    assert mask.sum() == k * 256
    return mask, np.sort(ix)


def random_scores(name):
    seed = int.from_bytes(hashlib.sha256(('region-v1:' + name).encode()).digest()[:8], 'little')
    return np.random.default_rng(seed).random(196)


def metrics(pred, target):
    pred, target = np.asarray(pred, bool), np.asarray(target, bool)
    tp = int(np.count_nonzero(pred & target))
    pp, gt = int(pred.sum()), int(target.sum())
    union = pp + gt - tp
    return dict(iou=tp / union if union else 1., dice=2 * tp / (pp + gt) if pp + gt else 1.,
                precision=tp / pp if pp else 0., recall=tp / gt if gt else 0.,
                fp=pp - tp, fn=gt - tp, label_pixels=gt)


def paired(values, repetitions=10000):
    a = np.asarray(values, dtype=np.float64)
    assert a.ndim == 1 and len(a) and np.isfinite(a).all()
    rng = np.random.default_rng(1219)
    out = []
    for start in range(0, repetitions, 200):
        ix = rng.integers(0, len(a), size=(min(200, repetitions-start), len(a)))
        out.extend(a[ix].mean(1).tolist())
    return dict(mean=float(a.mean()), ci95=np.quantile(out, [.025, .975]).tolist(), n=len(a))


def summarize(records):
    assert records and len({r['name'] for r in records}) == len(records)
    small = sorted(records, key=lambda r: (r['baseline']['label_pixels'], r['name']))[:(len(records)+3)//4]
    valid = [r for r in records if r['baseline']['fp'] + r['baseline']['fn'] > 0]
    assert valid
    cap_u = paired([r['selectors']['disagreement']['error_capture'] - r['selectors']['uncertainty']['error_capture'] for r in valid])
    cap_r = paired([r['selectors']['disagreement']['error_capture'] - r['selectors']['random']['error_capture'] for r in valid])
    correction = paired([r['selectors']['disagreement']['corrected']['iou'] - r['baseline']['iou'] for r in records])
    versus_u = paired([r['selectors']['disagreement']['corrected']['iou'] - r['selectors']['uncertainty']['corrected']['iou'] for r in records])
    delta_dice = float(np.mean([r['selectors']['disagreement']['corrected']['dice'] - r['baseline']['dice'] for r in records]))
    small_iou = float(np.mean([r['selectors']['disagreement']['corrected']['iou'] - r['baseline']['iou'] for r in small]))
    gates = dict(capture_beats_uncertainty_by_002=cap_u['mean'] >= .02,
                 capture_beats_random_by_005=cap_r['mean'] >= .05,
                 capture_uncertainty_ci_positive=cap_u['ci95'][0] > 0,
                 actual_iou_gain_at_least_001=correction['mean'] >= .001,
                 actual_iou_ci_positive=correction['ci95'][0] > 0,
                 dice_nondecreasing=delta_dice >= 0, small_area_iou_nondecreasing=small_iou >= 0,
                 beats_uncertainty_iou_by_0005=versus_u['mean'] >= .0005,
                 versus_uncertainty_iou_ci_positive=versus_u['ci95'][0] > 0)
    means = {name: {m: float(np.mean([r[name][m] for r in records])) for m in ('iou','dice','precision','recall','fp','fn')}
             for name in ('baseline', 'fine_probe', 'coarse_probe', 'full_correction')}
    for sel in SELECTORS:
        rows = [r['selectors'][sel] for r in records]
        means[sel] = dict(error_capture_mean=float(np.mean([r['selectors'][sel]['error_capture'] for r in valid])),
                         fp_capture_pooled=sum(r['fp_captured'] for r in rows)/max(1, sum(r['baseline']['fp'] for r in records)),
                         fn_capture_pooled=sum(r['fn_captured'] for r in rows)/max(1, sum(r['baseline']['fn'] for r in records)),
                         oracle_iou=float(np.mean([r['oracle_iou'] for r in rows])),
                         **{m: float(np.mean([r['corrected'][m] for r in rows])) for m in ('iou','dice','precision','recall','fp','fn')})
    return dict(samples=len(records), error_capture_eligible=len(valid), small_area_count=len(small),
                capture_minus_uncertainty=cap_u, capture_minus_random=cap_r, actual_iou_minus_r2=correction,
                actual_iou_minus_uncertainty=versus_u, dice_delta=delta_dice, small_area_iou_delta=small_iou,
                means=means, gates=gates, proceed_to_architecture=all(gates.values()),
                interpretation='Frozen-feature diagnostic; not a trained architecture result or Test score.')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('records', type=Path); p.add_argument('output', type=Path)
    args = p.parse_args(); data = json.loads(args.records.read_text(encoding='utf-8'))
    write_json(args.output, summarize(data['records']))

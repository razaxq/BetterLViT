"""CPU metrics, paired uncertainty and pre-registered progression gates."""
import hashlib
import json
from pathlib import Path
import numpy as np

ARMS = ('baseline', 'features', 'image')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)


def metrics(probability, label):
    pred, target = np.asarray(probability) > .5, np.asarray(label, bool)
    tp = int((pred & target).sum()); pp = int(pred.sum()); gt = int(target.sum())
    union = pp + gt - tp
    return dict(iou=tp/union if union else 1., dice=2*tp/(pp+gt) if pp+gt else 1.,
                precision=tp/pp if pp else 0., recall=tp/gt if gt else 0., fp=pp-tp, fn=gt-tp,
                label_pixels=gt, brier=float(np.square(np.asarray(probability, np.float64)-target).mean()))


def paired(values):
    a = np.asarray(values, dtype=np.float64)
    assert a.ndim == 1 and len(a) and np.isfinite(a).all()
    rng = np.random.default_rng(1219)
    samples = np.concatenate([a[rng.integers(0, len(a), (200, len(a)))].mean(1) for _ in range(50)])
    return dict(mean=float(a.mean()), ci95=np.quantile(samples, [.025,.975]).tolist(), n=len(a))


def summarize(records):
    assert records and len({r['name'] for r in records}) == len(records)
    keys = ('iou', 'dice', 'precision', 'recall', 'fp', 'fn', 'brier')
    means = {arm: {k: float(np.mean([r[arm][k] for r in records])) for k in keys} for arm in ARMS}
    small = sorted(records, key=lambda r: (r['baseline']['label_pixels'],r['name']))[:(len(records)+3)//4]
    delta = lambda rows, a, b, k: float(np.mean([r[a][k]-r[b][k] for r in rows]))
    vs_base = paired([r['image']['iou']-r['baseline']['iou'] for r in records])
    vs_control = paired([r['image']['iou']-r['features']['iou'] for r in records])
    control = paired([r['features']['iou']-r['baseline']['iou'] for r in records])
    safety = {k: delta(records, 'image','baseline',k) for k in ('dice','precision','brier')}
    safety.update({f'small_{k}':delta(small, 'image','baseline',k) for k in ('iou','dice','recall')})
    gates = dict(iou_vs_r2_at_least_003=vs_base['mean']>=.003, iou_vs_r2_ci_positive=vs_base['ci95'][0]>0,
                 iou_vs_features_at_least_001=vs_control['mean']>=.001, iou_vs_features_ci_positive=vs_control['ci95'][0]>0,
                 dice_nondecreasing=safety['dice']>=0, precision_nondecreasing=safety['precision']>=0,
                 small_dice_nondecreasing=safety['small_dice']>=0, small_recall_nondecreasing=safety['small_recall']>=0,
                 brier_nonincreasing=safety['brier']<=0)
    return dict(samples=len(records), small_count=len(small), means=means, image_minus_r2=vs_base,
                image_minus_features=vs_control, features_minus_r2=control, safety_deltas=safety,
                gates=gates, proceed=all(gates.values()), interpretation='Frozen-branch feasibility; no Test or novelty claim.')

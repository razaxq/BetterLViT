"""Audit completed P12 and diagnose paired errors from local artifacts only."""
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np

base = Path(__file__).resolve().parent
out = base / 'p12_results'
root = Path('D:/BetterLViT/outputs/visual_prior_20260908')
s = json.loads((out/'final_snapshot.json').read_text(encoding='utf-8'))
control = json.loads((base/'c4_validation.json').read_text(encoding='utf-8'))
candidate = s['files']['validation.json']
runtime = s['files']['runtime.json']
gate = json.loads((out/'c4_vs_p12.json').read_text(encoding='utf-8'))
state = json.loads((root/'p12_state.json').read_text(encoding='utf-8'))
forecast = json.loads((root/'p12_forecast.json').read_text(encoding='utf-8'))
assert runtime['phase'] == 'complete' and runtime['training_rc'] == runtime['validation_rc'] == 0
assert state['inspections_completed'] == 2 and not s['tracked_changes']
assert s['source_git_commit'] == runtime['source_git_commit'] == candidate['checkpoint_git_commit'] == candidate['analysis_git_commit'] == '5d09913d46863073cba93159af4ed61f88fdd96e'
assert not runtime['test_split_accessed'] and not candidate['test_split_accessed']
assert [x['epoch'] for x in s['epoch_timing']] == list(range(1,81))
assert 0 <= s['seconds_after_training'] <= 1800
assert gate['candidate_commit'] == s['source_git_commit'] and not gate['passed']
a = {r['name']: r for r in control['records']}
b = {r['name']: r for r in candidate['records']}
assert a.keys() == b.keys() and len(a) == len(b) == 1429
names = sorted(a)
areas = np.array([a[n]['label_pixels'] for n in names])
assert all(a[n]['label_pixels'] == b[n]['label_pixels'] for n in names)
cuts = np.quantile(areas, [.25, .5, .75])
groups = []
for number in range(4):
    selected = np.ones(len(names), dtype=bool)
    if number:
        selected &= areas > cuts[number-1]
    if number < 3:
        selected &= areas <= cuts[number]
    groups.append({'gt_area_quartile': number+1, 'samples': int(selected.sum()),
        'metrics': {key: {
            'control': float(np.mean([a[n][key] for n, yes in zip(names, selected) if yes])),
            'candidate': float(np.mean([b[n][key] for n, yes in zip(names, selected) if yes])),
            'delta': float(np.mean([b[n][key]-a[n][key] for n, yes in zip(names, selected) if yes]))}
            for key in ('iou','dice','precision','recall')}})


def totals(records):
    result = {'tp':0, 'fp':0, 'fn':0, 'predicted_pixels':0, 'label_pixels':0}
    for row in records.values():
        tp = round(row['recall'] * row['label_pixels'])
        assert abs(tp-row['recall']*row['label_pixels']) < 1e-7
        assert 0 <= tp <= min(row['label_pixels'], row['prediction_pixels'])
        if row['prediction_pixels']:
            assert abs(tp/row['prediction_pixels']-row['precision']) < 1e-12
        for key, value in {'tp':tp, 'fp':row['prediction_pixels']-tp,
            'fn':row['label_pixels']-tp, 'predicted_pixels':row['prediction_pixels'],
            'label_pixels':row['label_pixels']}.items():
            result[key] += value
    return result


history = {}
log = (out/'training.log').read_text(encoding='utf-8')
columns = ('train_loss','train_dice','train_iou','val_loss','val_dice','val_iou','lr')
for line in log.splitlines():
    parts = [x.strip() for x in line.split('|')]
    if len(parts) == 9 and parts[0].isdigit():
        row = dict(zip(columns, map(float, parts[1:8])))
        assert all(math.isfinite(value) for value in row.values())
        history[int(parts[0])] = dict(epoch=int(parts[0]), **row)
assert sorted(history) == list(range(1,81))
assert max(history, key=lambda epoch: history[epoch]['val_iou']) == candidate['checkpoint_best_epoch'] == 70
# Training log rows are rounded; the full-precision Best export is authoritative.
assert abs(history[70]['val_iou']-candidate['macro_iou']) < .000051
(out/'epoch_history_rounded.json').write_text(json.dumps(list(history.values()), indent=2)+'\n', encoding='utf-8')
deltas = np.array([b[n]['iou']-a[n]['iou'] for n in names])
at, bt = totals(a), totals(b)
local = lambda ts: datetime.fromtimestamp(ts,ZoneInfo('Australia/Sydney')).isoformat()
summary = {
    'source_git_commit': s['source_git_commit'], 'split':'validation', 'test_split_accessed':False,
    'status':'complete_gate_failed', 'epochs':80, 'best_epoch':70,
    'training_ended_sydney':local(runtime['training_ended_unix']),
    'validation_ended_sydney':local(runtime['validation_ended_unix']),
    'final_inspected_sydney':local(s['inspected_unix']),
    'seconds_after_training':s['seconds_after_training'], 'inspections_completed':2,
    'completion_prediction_error_seconds':runtime['training_ended_unix']-forecast['predicted_training_end_unix'],
    'control_metrics': {k:control[k] for k in ('macro_iou','macro_dice','macro_precision','macro_recall')},
    'candidate_metrics': {k:candidate[k] for k in ('macro_iou','macro_dice','macro_precision','macro_recall')},
    'gate_passed':False, 'size_quartile_cutoffs':cuts.tolist(), 'size_quartiles':groups,
    'iou_win_tie_loss':{'wins':int((deltas>0).sum()),'ties':int((deltas==0).sum()),'losses':int((deltas<0).sum())},
    'pooled_pixel_counts_diagnostic_only':{'control':at,'candidate':bt,
        'deltas':{key:bt[key]-at[key] for key in at}},
    'rounded_history_selected_epochs':[history[epoch] for epoch in (20,40,60,70,80)],
    'interpretation':'Lower precision and higher false-positive pixel count with only a small recall change are consistent with overprediction; this is not proof of the architectural cause.',
    'decision':'Stop this P12 configuration; do not launch C9, extend to 150, or evaluate Test.',
    'automation_deleted':True,
    'collector_console_issue':'SSH capture and UTF-8 file/state saves succeeded. Only final console print failed on cp1252 progress-bar glyphs; no SSH retry occurred. Console output is now ASCII escaped.',
}
(out/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
manifest = {p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    for p in out.iterdir() if p.is_file() and p.suffix in ('.json', '.log') and p.name != 'artifact_manifest.json'}
(out/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
state.update(phase='complete_gate_failed',gate_passed=False,automation_deleted=True,
    training_ended_unix=runtime['training_ended_unix'], completed_unix=runtime['completed_unix'],
    seconds_after_training=s['seconds_after_training'], best_checkpoint=runtime['best_checkpoint'])
(root/'p12_state.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))

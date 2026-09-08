"""Predict the final inspection from the saved first snapshot, without SSH."""
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

root = Path('D:/BetterLViT/outputs/visual_prior_20260908')
snapshot_path = root / 'p12_first_snapshot.json'
s = json.loads(snapshot_path.read_text(encoding='utf-8'))
state_path = root / 'p12_state.json'
state = json.loads(state_path.read_text(encoding='utf-8'))
runtime = s['files']['runtime.json']
assert state['inspections_completed'] == 1
assert runtime['phase'] == 'training' and s['launcher_process']
assert not s['tracked_changes'] and not runtime['test_split_accessed']
assert 'failure.json' not in s['files']
assert runtime['source_git_commit'] == s['source_git_commit'] == state['source_git_commit']
assert runtime['manifest'] == state['manifest']
rows = s['epoch_timing']
assert len(rows) >= 3
assert [x['epoch'] for x in rows] == list(range(1, len(rows) + 1))
assert all(x['source_git_commit'] == state['source_git_commit'] for x in rows)
durations = [x['duration_seconds'] for x in rows[1:]]
assert all(math.isfinite(x) and x > 0 for x in durations)
remaining = runtime['manifest']['epochs'] - rows[-1]['epoch']
assert remaining > 0
mean = statistics.mean(durations)
eta = rows[-1]['ended_unix'] + remaining * mean
check = math.ceil((eta + 12 * 60) / 60) * 60


def local(ts):
    return datetime.fromtimestamp(ts, ZoneInfo('Australia/Sydney')).isoformat()


forecast = {
    'source_git_commit': state['source_git_commit'],
    'snapshot_sha256': hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
    'inspected_unix': s['inspected_unix'],
    'inspected_sydney': local(s['inspected_unix']),
    'completed_epochs': len(rows), 'active_epoch': len(rows) + 1,
    'runtime_phase': runtime['phase'], 'inspections_completed': 1,
    'test_split_accessed': False, 'excluded_initial_epoch': 1,
    'measured_epoch_seconds': durations,
    'mean_epoch_seconds': mean,
    'median_epoch_seconds': statistics.median(durations),
    'remaining_epochs_from_last_completed_boundary': remaining,
    'predicted_training_end_unix': eta,
    'predicted_training_end_sydney': local(eta),
    'observed_range_extrapolation_sydney': [
        local(rows[-1]['ended_unix'] + remaining * x)
        for x in (min(durations), max(durations))],
    'planned_final_check_unix': check,
    'planned_final_check_sydney': local(check),
    'buffer_after_prediction_seconds': check - eta,
    'actual_training_end_window_for_30min_requirement_sydney': [local(check - 1800), local(check)],
    'assumptions': [
        'Future full epochs have approximately the mean duration of epochs 2-4.',
        'Normal epoch duration includes Train, Val and rolling checkpoint saves.',
        'Additional Best saves after epoch 5 and throughput drift consume the 12-minute buffer.',
        'Observed-range extrapolation is descriptive, not a predictive confidence interval.',
        'The final snapshot must verify actual finish-to-inspection delay; prediction is not a guarantee.',
    ],
}
(root / 'p12_forecast.json').write_text(json.dumps(forecast, indent=2) + '\n', encoding='utf-8')
state.update(dispatch_only=False, health_confirmed=True, phase='training', forecast=forecast,
             planned_final_check_unix=check, planned_final_check_sydney=local(check))
state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
print(json.dumps(forecast, indent=2))

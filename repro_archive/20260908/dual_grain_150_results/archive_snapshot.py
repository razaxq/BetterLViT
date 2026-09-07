"""Verify and archive the existing final snapshot locally, without a connection."""
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT = Path('D:/BetterLViT/outputs/dual_grain_150_monitor_20260907')
DEST = Path(__file__).resolve().parent
PARENT = DEST.parents[1] / '20260907/dual_grain_results'
SHA = 'c724a62001f6c3b809cb12eee78f3bee79bdb60a'
PARENT_SHA = '2fc6ab5c8e4662d741fd8b994e55b780391948ac'


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def main():
    snapshot = json.loads((ROOT / 'final_inspection.json').read_text(encoding='utf-8'))
    assert snapshot['repository_commit'] == SHA and not snapshot['tracked_source_changes']
    statuses = snapshot['status']
    assert statuses['chain']['value'] == 'complete'
    assert all(v['value'] == '0' for k, v in statuses.items() if k != 'chain')
    files = snapshot['files']
    runtime, selection = files['runtime.json'], files['selection.json']
    assert runtime['source_git_commit'] == SHA and runtime['parent_source_git_commit'] == PARENT_SHA
    assert runtime['total_epochs'] == selection['completed_epochs'] == 150
    assert selection['best_epoch'] == 80
    assert selection['checkpoint'] == runtime['parent_best_checkpoint']
    assert selection['checkpoint_sha256'] == runtime['parent_best_sha256']
    assert selection['selection_scope'] == 'all 150 validation epochs'
    for split, samples in [('test', 2113), ('validation', 1429)]:
        result = files[f'p11_{split}.json']
        previous = json.loads((PARENT / f'p11_{split}.json').read_text(encoding='utf-8'))
        comparison = files[f'p11_80_vs_150_{split}.json']
        assert result['checkpoint_git_commit'] == result['analysis_git_commit'] == PARENT_SHA
        assert result['checkpoint'] == selection['checkpoint']
        assert result['checkpoint_best_epoch'] == 80 and result['epochs'] == 80
        assert result['threshold'] == 0.5 and result['seed'] == 1219
        assert result['samples'] == len(result['records']) == samples
        assert len({r['name'] for r in result['records']}) == samples
        assert result['records'] == previous['records']
        assert comparison['completed_epochs'] == 150 and comparison['best_epoch'] == 80
        for metric in ('iou', 'dice', 'precision', 'recall'):
            value = result[f'macro_{metric}']
            assert math.isclose(mean(r[metric] for r in result['records']), value, abs_tol=1e-12)
            row = comparison['metrics'][metric]
            assert row['parent_80'] == row['continued_150'] == value
            assert row['delta'] == 0
    finished = statuses['training']['modified_unix']
    inspected = snapshot['inspected_unix']
    delay = inspected - finished
    assert 0 <= delay <= 1800
    state = json.loads((ROOT / 'state.json').read_text(encoding='utf-8'))
    state.update(final_inspection_completed=True, inspections_completed=2,
        final_inspection_utc=datetime.fromtimestamp(inspected, timezone.utc).isoformat(),
        training_finished_utc=datetime.fromtimestamp(finished, timezone.utc).isoformat(),
        training_end_time_source='successful training.status modification time',
        final_inspection_delay_seconds=delay, final_after_training_requirement_met=True,
        final_inspection_within_30_minutes=True, all_stages_completed=True,
        completed_epochs=150, best_epoch=80, outcome='retained_epoch_80_best_no_gain',
        automation_status='PAUSED')
    save(ROOT / 'state.json', state)
    save(DEST / 'monitor_final_state.json', state)
    for name, value in files.items():
        save(DEST / name, value)
    save(DEST / 'inspection_proof.json', {k: v for k, v in snapshot.items() if k != 'files'})
    print(json.dumps({'verified': True, 'completed_epochs': 150, 'best_epoch': 80,
                      'test_records_identical_to_parent': True, 'inspection_delay_seconds': delay}))


if __name__ == '__main__':
    main()

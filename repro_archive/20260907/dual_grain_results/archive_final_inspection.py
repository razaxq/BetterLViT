"""Verify and archive the existing snapshot locally; never connects to the server."""
import ast
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

SOURCE = Path('D:/BetterLViT/outputs/dual_grain_monitor_20260907')
DEST = Path(__file__).resolve().parent
SHA = '2fc6ab5c8e4662d741fd8b994e55b780391948ac'


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def utc(stamp):
    return datetime.fromtimestamp(stamp, timezone.utc).isoformat()


def main():
    snapshot = json.loads((SOURCE / 'final_inspection.json').read_text(encoding='utf-8'))
    assert snapshot['repository_commit'] == SHA
    assert snapshot['tracked_source_changes'] == ''
    status = snapshot['status']
    assert status['chain']['value'] == 'complete'
    assert all(v['value'] == '0' for k, v in status.items() if k != 'chain')
    files = snapshot['files']
    assert files['runtime.json']['source_git_commit'] == SHA
    for split, count in [('test', 2113), ('validation', 1429)]:
        result = files[f'p11_{split}.json']
        comparison = files[f'c4_vs_p11_{split}.json']
        assert result['checkpoint_git_commit'] == result['analysis_git_commit'] == SHA
        assert comparison['candidate_commit'] == SHA
        assert comparison['control_commit'] == 'add4908a0d6f702b0a10c4581725b535543829b8'
        assert result['samples'] == len(result['records']) == comparison['samples'] == count
        assert len({r['name'] for r in result['records']}) == count
        assert result['threshold'] == comparison['threshold'] == 0.5
        assert result['seed'] == 1219 and result['epochs'] == 80
        assert result['checkpoint_best_epoch'] == 80
        for metric in ('iou', 'dice', 'precision', 'recall'):
            aggregate = result[f'macro_{metric}']
            assert math.isclose(mean(r[metric] for r in result['records']), aggregate, abs_tol=1e-12)
            row = comparison['metrics'][metric]
            assert math.isclose(row['candidate'], aggregate, abs_tol=1e-12)
            assert math.isclose(row['candidate'] - row['control'], row['delta'], abs_tol=1e-12)
    assert files['p11_test.json']['checkpoint'] == files['p11_validation.json']['checkpoint']
    assert files['p11_test.json']['threshold_selected_on_test'] is False
    finished = status['training']['modified_unix']
    inspected = snapshot['inspected_unix']
    delay = inspected - finished
    assert 0 <= delay <= 1800
    state = json.loads((SOURCE / 'state.json').read_text(encoding='utf-8'))
    state.update(final_inspection_completed=True, inspections_completed=2,
                 final_inspection_utc=utc(inspected), training_finished_utc=utc(finished),
                 training_end_time_source='training.status modification time',
                 final_inspection_delay_seconds=delay,
                 final_after_training_requirement_met=True,
                 final_inspection_within_30_minutes=True,
                 all_stages_completed=True, outcome='no_test_iou_gain',
                 automation_status='PAUSED')
    write_json(SOURCE / 'state.json', state)
    write_json(DEST / 'monitor_final_state.json', state)
    for name, data in files.items():
        write_json(DEST / name, data)
    proof = {k: v for k, v in snapshot.items() if k != 'files'}
    write_json(DEST / 'inspection_proof.json', proof)
    collector = SOURCE / 'collect_final_inspection.py'
    ast.parse(collector.read_text(encoding='utf-8'))
    shutil.copy2(collector, DEST / collector.name)
    print(json.dumps({'verified': True, 'delay_seconds': delay, 'archived_files': len(files)}))


if __name__ == '__main__':
    main()

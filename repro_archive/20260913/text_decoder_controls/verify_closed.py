"""Verify completed artifacts and native automation deletion without remote polling."""
import json
import time
from pathlib import Path

from remote_ops import HERE, read, save

receipt = read(HERE / 'automation_deletion_receipt.json')
assert receipt['automationId'] == 'betterlvit' and receipt['deleteStatus'] == 'deleted'
config = Path('C:/Users/dtftn/.codex/automations/betterlvit/automation.toml')
assert not config.exists(), 'The completed-batch automation still has a local configuration'
proofs = {}
for label in ('t1', 't2'):
    state = read(HERE / (label + '_state.json'))
    assert state['last_phase'] == 'complete' and state['inspections_completed'] == 2
    folder = HERE / (label + '_results')
    for name in ('independent_verification.json', 'hf_upload_verified.json', 'download_xet_verified.json'):
        assert read(folder / name)['verified']
    proof = read(folder / 'independent_verification.json')
    assert not proof['threshold_reconciliation_needed'] and proof['within_30_minutes']
    proofs[label] = dict(source_git_commit=state['source_git_commit'], inspections_completed=2,
                        seconds_after_training_end=proof['seconds_after_training_end'])
result = dict(verified=True, checked_unix=time.time(), automation_id='betterlvit',
              native_delete_status=receipt['deleteStatus'], automation_config_absent=True,
              completed_runs=proofs, new_training_started=False, test_split_accessed=False)
save('batch_closed_verified.json', result)
print(json.dumps(result))

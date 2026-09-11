"""Read-only confirmation of the native scheduled appointment; never edits TOML."""
from datetime import datetime
import json
from pathlib import Path
import tomllib
from analysis import write_json
HERE=Path(__file__).resolve().parent
state=json.loads((HERE/'state.json').read_text())
data=tomllib.loads(Path('C:/Users/dtftn/.codex/automations/betterlvit/automation.toml').read_text(encoding='utf-8'))
appointment=datetime.fromisoformat(state['planned_check_sydney'])
expected=f'FREQ=DAILY;BYHOUR={appointment.hour};BYMINUTE={appointment.minute};BYSECOND=0'
assert data['id']=='betterlvit' and data['kind']=='heartbeat' and data['status']=='ACTIVE'
assert data['rrule']==expected and data['target_thread_id']=='01a074f2-5c58-7221-9a58-59850e626880'
assert state['source_git_commit'] in data['prompt'] and 'control.py train_collect' in data['prompt']
value=dict(verified=True,automation_id=data['id'],kind=data['kind'],status=data['status'],
    target_thread_id=data['target_thread_id'],planned_check_sydney=state['planned_check_sydney'],
    source_git_commit=state['source_git_commit'],inspections_completed=state['inspections_completed'],maximum_inspections=2,
    existing_native_automation_updated=True,toml_read_only=True)
write_json(HERE/'automation_verified.json',value);print(json.dumps(value))

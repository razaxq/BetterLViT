"""Verify the locally saved native heartbeat; never contacts the training server."""
from datetime import datetime
import json
from pathlib import Path
import time
import tomllib
from zoneinfo import ZoneInfo
from analysis import digest,write_json

HERE=Path(__file__).resolve().parent
PATH=Path('C:/Users/dtftn/.codex/automations/betterlvit/automation.toml')
saved=tomllib.loads(PATH.read_text(encoding='utf-8'))
request=json.loads((HERE/'first_check_appointment_request.json').read_text())
state=json.loads((HERE/'state.json').read_text())
for key in ('id','kind','name','status','rrule','prompt'):assert saved[key]==request[key],key
assert saved['target_thread_id']==request['targetThreadId']
when=datetime.fromtimestamp(state['planned_check_unix'],ZoneInfo('Australia/Sydney'))
assert when.isoformat()==state['planned_check_sydney']
assert saved['rrule']==f'FREQ=DAILY;BYHOUR={when.hour};BYMINUTE={when.minute};BYSECOND=0'
proof=dict(verified=True,automation_id=saved['id'],kind=saved['kind'],status=saved['status'],
    target_thread_id=saved['target_thread_id'],next_check_sydney=when.isoformat(),
    source_git_commit=state['source_git_commit'],inspections_completed=state['inspections_completed'],
    automation_toml_sha256=digest(PATH),prompt_exact_match=True,verified_unix=time.time(),
    remote_training_status_queried=False)
write_json(HERE/'first_check_appointment_verified.json',proof)
print(json.dumps(proof))

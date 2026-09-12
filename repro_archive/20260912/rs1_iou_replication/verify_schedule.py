"""Verify the native heartbeat's saved settings without querying training."""
import argparse,json,tomllib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from remote_ops import HERE,read,save

def main():
    p=argparse.ArgumentParser();p.add_argument('--label',choices=('rs1s2027','rs1s3407'),required=True)
    p.add_argument('--phase',choices=('first','final'),required=True)
    p.add_argument('--output',default='automation_launch_verified.json');a=p.parse_args()
    config=tomllib.loads(Path('C:/Users/dtftn/.codex/automations/betterlvit/automation.toml').read_text(encoding='utf-8'))
    state=read(HERE/(a.label+'_state.json'));key='planned_first_check_unix' if a.phase=='first' else 'final_check_unix'
    planned=state[key];local=datetime.fromtimestamp(planned,ZoneInfo('Australia/Sydney'))
    expected=f'FREQ=DAILY;BYHOUR={local.hour};BYMINUTE={local.minute};BYSECOND=0'
    assert config['kind']=='heartbeat' and config['status']=='ACTIVE' and config['id']=='betterlvit'
    assert config['rrule']==expected
    assert config['target_thread_id']=='01a074f2-5c58-7221-9a58-59850e626880'
    registered=(HERE/'heartbeat_prompt.txt').read_text(encoding='utf-8')
    assert config['prompt'].rstrip('\n')==registered.rstrip('\n'),'Only the app-trimmed trailing newline is normalized'
    value=dict(verified=True,automation_id=config['id'],status=config['status'],kind=config['kind'],rrule=config['rrule'],
        target_thread_id=config['target_thread_id'],label=a.label,phase=a.phase,planned_check_sydney=local.isoformat(),
        planned_check_unix=planned,inspections_completed=state['inspections_completed'],source_git_commit=state['source_git_commit'],
        prompt_matches_registered_workflow=True,app_trims_trailing_newline=True)
    save(a.output,value);print(json.dumps(value))

if __name__=='__main__':main()

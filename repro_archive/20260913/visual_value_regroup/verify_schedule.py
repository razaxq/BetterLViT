"""Verify native app scheduling against local experiment state; no server call."""
import argparse,tomllib,json
from pathlib import Path
from remote_ops import HERE,read,save
p=argparse.ArgumentParser();p.add_argument('--label',choices=('m1','m2'),required=True)
p.add_argument('--phase',choices=('first','final'),required=True);a=p.parse_args()
expected=read(HERE/'automation_update_request.json');identity=read(HERE/'automation_identity.json')
actual=tomllib.loads(Path(identity['config_path']).read_text(encoding='utf-8'))
assert actual['id']==identity['id'] and actual['target_thread_id']==identity['target_thread_id']
for key in ('rrule','kind','status','name','prompt'):assert actual[key]==expected[key],key
state=read(HERE/(a.label+'_state.json'))
key='planned_first_check_unix' if a.phase=='first' else 'final_check_unix'
proof=dict(verified=True,label=a.label,phase=a.phase,automation_id=actual['id'],
    target_thread_id=actual['target_thread_id'],planned_check_unix=state[key],rrule=actual['rrule'],
    inspections_completed=state['inspections_completed'],source_git_commit=state['source_git_commit'])
save(a.label+'_automation_'+a.phase+'_verified.json',proof);print(json.dumps(proof))

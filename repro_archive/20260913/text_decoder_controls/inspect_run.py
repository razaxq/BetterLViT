"""Exactly one bounded status snapshot per call, enforcing two checks per run."""
import argparse
from datetime import datetime
import json
import math
import time
from zoneinfo import ZoneInfo
from remote_ops import HERE,remote,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--label',choices=('t1','t2'),required=True)
    p.add_argument('--phase',choices=('first','final'),required=True);args=p.parse_args()
    state=json.loads((HERE/(args.label+'_state.json')).read_text())
    assert state['inspections_completed']<2,'Training inspection budget exhausted'
    if args.phase=='first':assert state['inspections_completed']==0
    else:assert state['inspections_completed']==1
    planned=state['planned_first_check_unix'] if args.phase=='first' else state['final_check_unix']
    assert time.time()>=planned, 'Wait for the predicted inspection time; do not poll early'
    state['inspections_completed']+=1
    state['inspection_attempt_started_unix']=time.time()
    save(args.label+'_state.json',state)
    source=json.loads((HERE/'sources.json').read_text())[args.label]
    value=remote('SOURCE='+repr(source)+'\n'+'''
import json,subprocess,time
from pathlib import Path
run=Path(SOURCE['remote_run']);repo=Path(SOURCE['repository'])
runtime=json.loads((run/'runtime.json').read_text())
assert runtime['source_git_commit']==SOURCE['source_git_commit']
timing=[json.loads(line) for line in (run/'epoch_timing.jsonl').read_text().splitlines()] if (run/'epoch_timing.jsonl').exists() else []
value=dict(checked_unix=time.time(),runtime=runtime,timing=timing,
    training_log_tail=(run/'training.log').read_text(errors='replace')[-5000:],
    runtime_source_clean=not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip(),
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip())
if runtime['phase']=='complete':
    v=json.loads((run/'validation.json').read_text())
    value['validation_summary']={k:v[k] for k in ('samples','macro_iou','macro_dice','macro_precision','macro_recall','macro_brier','checkpoint_best_epoch','checkpoint_git_commit')}
print(json.dumps(value))
''')
    value['inspection_number']=state['inspections_completed']
    phase=value['runtime']['phase']
    if phase=='complete':
        value['seconds_after_training_end']=value['checked_unix']-value['runtime']['training_ended_unix']
        value['within_30_minutes']=value['seconds_after_training_end']<=1800
    elif phase=='training' and args.phase=='first':
        usable=[r for r in value['timing'] if r['epoch']>1]
        if not usable:
            raise RuntimeError('No complete post-warmup epoch: preserve this check and investigate failure without repeated status polling')
        if usable:
            rate=sum(r['duration_seconds'] for r in usable[-3:])/len(usable[-3:])
            last=value['timing'][-1]
            end=last['ended_unix']+(80-last['epoch'])*rate
            value['forecast']=dict(mean_epoch_seconds=rate,predicted_training_end_unix=end,
                final_check_unix=math.ceil((end+720)/60)*60,remaining_audit_allowance_seconds=0)
            state.update(value['forecast'])
    for k in ('checked_unix',):value[k.replace('_unix','_sydney')]=datetime.fromtimestamp(value[k],ZoneInfo('Australia/Sydney')).isoformat()
    if 'forecast' in value:
        for k in ('predicted_training_end_unix','final_check_unix'):
            value['forecast'][k.replace('_unix','_sydney')]=datetime.fromtimestamp(value['forecast'][k],ZoneInfo('Australia/Sydney')).isoformat()
    state['last_phase']=phase;state['last_checked_unix']=value['checked_unix']
    save(args.label+'_'+args.phase+'_snapshot.json',value);save(args.label+'_state.json',state)
    print(json.dumps({k:v for k,v in value.items() if k not in ('training_log_tail','timing')},indent=2))


if __name__=='__main__':main()

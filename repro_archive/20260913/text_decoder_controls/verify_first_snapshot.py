"""Verify a saved first snapshot without making another remote status call."""
import argparse
import math
import re
from pathlib import Path
from remote_ops import HERE, read, save
p=argparse.ArgumentParser();p.add_argument('--label',choices=('t1','t2'),required=True);label=p.parse_args().label
source=read(HERE/'sources.json')[label]
snapshot=read(HERE/(label+'_first_snapshot.json'))
state=read(HERE/(label+'_state.json'))
manifest=read(Path(source['local_repository'])/'experiment_manifests/active_decoder.json')
assert snapshot['runtime']['manifest']==manifest
assert snapshot['runtime']['source_git_commit']==state['source_git_commit']==source['source_git_commit']
assert snapshot['runtime_source_clean'] and snapshot['runtime']['phase']=='training'
assert not snapshot['runtime']['test_split_accessed']
assert snapshot['inspection_number']==state['inspections_completed']==1
rows=snapshot['timing'];assert [r['epoch'] for r in rows]==list(range(1,len(rows)+1))
assert all(r['source_git_commit']==source['source_git_commit'] and r['epochs']==80 for r in rows)
assert all(math.isfinite(r['duration_seconds']) and r['duration_seconds']>0 for r in rows)
tail=snapshot['training_log_tail']
assert not any(x in tail for x in ('Traceback (most recent call last)','CUDA out of memory','RuntimeError:','FloatingPointError:'))
progress=re.findall(r'\[Train\] Epoch: \[(\d+)\]\[(\d+)/(\d+)\]',tail)
latest=list(map(int,progress[-1])) if progress else None
forecast=snapshot['forecast'];usable=[r for r in rows if r['epoch']>1][-3:]
rate=sum(r['duration_seconds'] for r in usable)/len(usable)
assert rate==forecast['mean_epoch_seconds']
assert forecast['predicted_training_end_unix']==rows[-1]['ended_unix']+(80-rows[-1]['epoch'])*rate
assert forecast['final_check_unix']==state['final_check_unix']
assert 720<=forecast['final_check_unix']-forecast['predicted_training_end_unix']<780
proof=dict(verified=True,label=label,source_git_commit=source['source_git_commit'],
    checked_sydney=snapshot['checked_sydney'],completed_epochs=len(rows),latest_logged_train_progress=latest,
    gpu=snapshot['gpu'],inspection_number=1,source_and_manifest_match=True,log_tail_no_fatal_error=True,
    forecast=forecast,formal_result_available=False,test_split_accessed=False,
    scope='Only the saved first inspection; no second server access')
save(label+'_first_verified.json',proof)
print(proof)

"""Verify saved completion and compare matching seeds locally only."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from prepare_replication import HERE,ROOT

p=argparse.ArgumentParser()
p.add_argument('--label',choices=('c4s2027','r2s2027','c4s3407','r2s3407'),required=True)
label=p.parse_args().label
sources=json.loads((HERE/'sources.json').read_text())
s=sources[label]
state_path=ROOT/f'{label}_state.json'
state=json.loads(state_path.read_text(encoding='utf-8'))
snap=json.loads((ROOT/f'{label}_final_snapshot.json').read_text(encoding='utf-8'))
assert 1<=state['inspections_completed']<=2
assert not snap['tracked_changes']
runtime=snap['files']['runtime.json']
assert runtime['source_git_commit']==snap['source_git_commit']==s['source_git_commit']
assert runtime['manifest']==state['manifest'] and not runtime['test_split_accessed']
delay=snap.get('seconds_after_training')
state.update(phase=runtime['phase'],seconds_after_training=delay,final_check_within_30min=delay is not None and 0<=delay<=1800)
state_path.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8',newline='\n')
if runtime['phase']!='complete':
    print(json.dumps(dict(phase=runtime['phase'],error=runtime.get('error'),instruction='Do not add a third inspection or continue the chain until this run is resolved.')))
    sys.exit(0)
assert runtime['training_rc']==runtime['validation_rc']==0
assert [r['epoch'] for r in snap['epoch_timing']]==list(range(1,81))
assert all(r['source_git_commit']==s['source_git_commit'] for r in snap['epoch_timing'])
value=snap['files']['validation.json']
assert value['seed']==s['seed'] and value['experiment']==s['profile'] and value['epochs']==80 and value['samples']==1429
assert value['checkpoint_git_commit']==value['analysis_git_commit']==s['source_git_commit']
assert value['split']=='validation' and not value['test_split_accessed'] and value['threshold']==.5 and value['selection_metric']=='iou'
assert not any(value[k] for k in ('text_use_lora','boundary_loss_weight','bcdh_enabled','cdrr_enabled','race_enabled','race_pe_enabled'))
for metric in ('iou','dice','precision','recall'):
    assert abs(sum(r[metric] for r in value['records'])/1429-value['macro_'+metric])<1e-12
path=ROOT/f'{label}_validation.json'
path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')
summary=dict(label=label,source_git_commit=s['source_git_commit'],seed=s['seed'],role=s['role'],phase='complete',
    epochs=80,best_epoch=value['checkpoint_best_epoch'],macro_iou=value['macro_iou'],macro_dice=value['macro_dice'],
    seconds_after_training=delay,final_check_within_30min=state['final_check_within_30min'],inspections_completed=state['inspections_completed'],test_split_accessed=False)
if s['role']=='r2':
    control=ROOT/('c4s'+str(s['seed'])+'_validation.json')
    gate=ROOT/('c4_vs_r2_seed'+str(s['seed'])+'.json')
    subprocess.run([sys.executable,str(Path(s['local_repository'])/'tools/compare_recipe_validation.py'),
        '--control',str(control),'--candidate',str(path),'--manifest',str(ROOT/f'{label}_manifest.json'),'--output',str(gate)],check=True)
    summary['paired_gate']=json.loads(gate.read_text())
    summary['continuation']='Complete both predeclared additional seeds regardless of this individual gate.'
(ROOT/f'{label}_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8',newline='\n')
state['summary']=summary
state_path.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(summary))

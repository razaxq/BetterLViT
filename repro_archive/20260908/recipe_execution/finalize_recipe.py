"""Analyze a saved final snapshot locally; never connects to the training server."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from prepare_recipe import HERE, ROOT


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--label',choices=('r1','r2'),required=True)
    label=p.parse_args().label
    source=json.loads((HERE/'sources.json').read_text())[label]
    state_path=ROOT/f'{label}_state.json'
    state=json.loads(state_path.read_text(encoding='utf-8'))
    snap=json.loads((ROOT/f'{label}_final_snapshot.json').read_text(encoding='utf-8'))
    assert 1 <= state['inspections_completed'] <= 2
    assert not snap['tracked_changes']
    runtime=snap['files']['runtime.json']
    assert snap['source_git_commit']==runtime['source_git_commit']==source['source_git_commit']
    assert runtime['manifest']==state['manifest']
    assert not runtime['test_split_accessed']
    state.update(phase=runtime['phase'],final_snapshot=state['last_snapshot'])
    delay=snap.get('seconds_after_training')
    state.update(seconds_after_training=delay,
        final_check_within_30min=delay is not None and 0<=delay<=1800)
    state_path.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    if runtime['phase']!='complete':
        print(json.dumps(dict(phase=runtime['phase'],error=runtime.get('error'),inspections_completed=state['inspections_completed'],
            instruction='Do not consume a third inspection or launch another training while this run remains active.')))
        return
    assert runtime['training_rc']==0 and runtime['validation_rc']==0
    rows=snap['epoch_timing']
    assert [r['epoch'] for r in rows]==list(range(1,81))
    assert all(r['source_git_commit']==source['source_git_commit'] for r in rows)
    result=snap['files']['validation.json']
    assert result['checkpoint_git_commit']==result['analysis_git_commit']==source['source_git_commit']
    assert result['epochs']==80 and result['samples']==1429
    candidate=ROOT/f'{label}_validation.json'
    candidate.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    gate=ROOT/f'c4_vs_{label}.json'
    subprocess.run([sys.executable,str(Path(source['local_repository'])/'tools/compare_recipe_validation.py'),
        '--control',str(HERE.parent/'visual_prior_execution/c4_validation.json'),
        '--candidate',str(candidate),'--manifest',str(ROOT/f'{label}_manifest.json'),
        '--output',str(gate)],check=True)
    decision=json.loads(gate.read_text())
    summary=dict(label=label,phase='complete',source_git_commit=source['source_git_commit'],
        test_split_accessed=False,epochs=80,best_epoch=result['checkpoint_best_epoch'],
        macro_iou=result['macro_iou'],macro_dice=result['macro_dice'],
        gate_passed=decision['passed'],gate_checks=decision['checks'],deltas=decision['deltas'],
        inspections_completed=state['inspections_completed'],seconds_after_training=delay,
        final_check_within_30min=state['final_check_within_30min'])
    (ROOT/f'{label}_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    state.update(summary=summary)
    state_path.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary))


if __name__=='__main__':
    main()

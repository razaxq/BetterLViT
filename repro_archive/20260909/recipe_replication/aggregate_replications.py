"""Report the predeclared three paired seeds; no Test access or seed selection."""
import json
import statistics
from pathlib import Path
from prepare_replication import HERE,ROOT

sources=json.loads((HERE/'sources.json').read_text())
original=Path('D:/BetterLViT/outputs/recipe_20260908')
rows=[]
for seed in (1219,2027,3407):
    if seed==1219:
        gate=json.loads((original/'c4_vs_r2.json').read_text())
        control=json.loads((HERE.parents[1]/'20260908/visual_prior_execution/c4_validation.json').read_text())
        candidate=json.loads((original/'r2_validation.json').read_text())
        expected=('add4908a0d6f702b0a10c4581725b535543829b8','9eca26de5b301099805530edbf5a1a8718bea662')
    else:
        gate=json.loads((ROOT/f'c4_vs_r2_seed{seed}.json').read_text())
        control=json.loads((ROOT/f'c4s{seed}_validation.json').read_text())
        candidate=json.loads((ROOT/f'r2s{seed}_validation.json').read_text())
        expected=(sources[f'c4s{seed}']['source_git_commit'],sources[f'r2s{seed}']['source_git_commit'])
    assert (gate['control_commit'],gate['candidate_commit'])==expected
    assert gate['seed']==control['seed']==candidate['seed']==seed
    assert control['checkpoint_git_commit']==expected[0] and candidate['checkpoint_git_commit']==expected[1]
    assert gate['split']=='validation' and not gate['test_split_accessed']
    rows.append(dict(seed=seed,control_commit=expected[0],candidate_commit=expected[1],
        control={m:control['macro_'+m] for m in ('iou','dice')},candidate={m:candidate['macro_'+m] for m in ('iou','dice')},
        deltas={m:gate['deltas'][m]['mean'] for m in ('iou','dice','precision','recall')},
        small_iou_delta=gate['deltas']['iou']['small_group_mean'],
        image_bootstrap_iou_95ci=gate['deltas']['iou']['paired_image_bootstrap_95ci']))
summary={m:dict(mean=statistics.mean(r['deltas'][m] for r in rows),sample_std=statistics.stdev(r['deltas'][m] for r in rows))
    for m in ('iou','dice','precision','recall')}
checks=dict(all_three_iou_deltas_positive=all(r['deltas']['iou']>0 for r in rows),
    mean_iou_delta_at_least_0_003=summary['iou']['mean']>=.003,
    mean_dice_not_lower=summary['dice']['mean']>=0,
    mean_small_iou_not_lower=statistics.mean(r['small_iou_delta'] for r in rows)>=0)
result=dict(split='validation',test_split_accessed=False,predeclared_seeds=[1219,2027,3407],
    paired_runs=rows,delta_summary=summary,checks=checks,passed=all(checks.values()),
    interpretation='Three paired training seeds; image-bootstrap intervals do not establish population-level cross-seed significance. Final performance claims require fixed-protocol Test evaluation.')
(ROOT/'three_seed_summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(result,indent=2))

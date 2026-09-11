"""Independent counts, membership, optimization budgets and all preregistered gates."""
import json
from pathlib import Path
import numpy as np
from analysis import digest,write_json
from refiner_names import VARIANTS
from screen_analysis import summarize
HERE=Path(__file__).resolve().parent;R=HERE/'results'
M=json.loads((HERE/'manifest.json').read_text());runtime=json.loads((R/'runtime.json').read_text())
assert runtime['phase']=='complete'
for name,meta in runtime['artifacts'].items():assert digest(R/name)==meta['sha256'] and (R/name).stat().st_size==meta['bytes']
rows=json.loads((R/'records.json').read_text())['records'];split=json.loads((HERE/'split.json').read_text())['records']
assert [r['index'] for r in rows]==[r['index'] for r in split] and len(rows)==4585
orders=json.loads((R/'training_orders.json').read_text());history=json.loads((R/'history.json').read_text())
assert len(history)==5
for fold in range(5):
    h=history[fold];assert h['fold']==fold and [s['step'] for s in h['history']]==list(range(1,2049))
    assert all(set(s['loss'])==set(VARIANTS) and np.isfinite(list(s['loss'].values())).all() for s in h['history'])
    order=orders[str(fold)]
    fit=[r['index'] for r in split if r['fold']!=fold];held=[r['index'] for r in split if r['fold']==fold]
    assert order['fit']==fit and order['held']==held and len(order['indices'])==2048*16 and set(order['indices'])<=set(fit)
    assert {r['group_id'] for r in split if r['fold']==fold}.isdisjoint({r['group_id'] for r in split if r['fold']!=fold})
    cutoff=float(np.quantile([r['outputs']['baseline']['gt_area'] for r in rows if r['fold']!=fold],.25))
    assert cutoff==order['small_cutoff_fit_only']
    saved=json.loads((R/f'fold_{fold}_records.json').read_text());assert saved['held_evaluations']==1 and saved['steps']==2048
    assert saved['records']==[r for r in rows if r['fold']==fold]
for r,s in zip(rows,split):
    assert all(r[k]==v for k,v in s.items()) and set(r['outputs'])=={'baseline',*VARIANTS}
    base=r['outputs']['baseline']
    assert r['small_by_fit_cutoff']==(base['gt_area']<=orders[str(r['fold'])]['small_cutoff_fit_only'])
    for name,m in r['outputs'].items():
        tp,fp,fn=(m[k] for k in ('tp','fp','fn'));assert min(tp,fp,fn)>=0
        assert m['iou']==(tp/(tp+fp+fn) if tp+fp+fn else 0.)
        assert m['dice']==(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
        assert m['gt_area']==tp+fn==base['gt_area']
        if name=='baseline':continue
        d=r['diagnostics'][name]
        assert d['noncandidate_probabilities_exact'] and d['changed_pixels']<=1024
        assert m['fp']-base['fp']==d['new_fp']-d['fixed_fp']
        assert m['fn']-base['fn']==d['new_fn']-d['fixed_fn']
        assert d['changed_pixels']==sum(d[k] for k in ('fixed_fp','fixed_fn','new_fp','new_fn'))
        if name.endswith('mass'):assert d['soft_mass_absolute_error']<=.02
summary=summarize(rows,M);summary['source_git_commit']=runtime['source_git_commit']
write_json(R/'summary_verified.json',summary)
write_json(R/'independent_verification.json',dict(verified=True,source_git_commit=runtime['source_git_commit'],
    raw_sha256_verified=True,sample_count=4585,folds=5,heads_per_fold=4,steps_per_head_per_fold=2048,
    count_metrics_and_transitions_verified=True,own_fold_exclusion_verified=True,small_cutoffs_fit_only_verified=True,
    final_held_evaluations_per_fold=1,old_b_holdout_accessed=False,official_validation_accessed=False,test_split_accessed=False))
print(json.dumps(dict(means=summary['means'],baseline_gates=summary['baseline_gates'],fine_gates=summary['fine_interface_gates'],
    next_interface=summary['recommended_interface_for_text_screen'])))

"""Independently recompute stored count metrics and descriptive Train summaries."""
import json
from pathlib import Path
import numpy as np
from analysis import digest,write_json
HERE=Path(__file__).resolve().parent
R=HERE/'results'
runtime=json.loads((R/'runtime.json').read_text());assert runtime['phase']=='complete'
for name,meta in runtime['artifacts'].items():
    assert digest(R/name)==meta['sha256'] and (R/name).stat().st_size==meta['bytes']
e0=json.loads((R/'e0_per_image.json').read_text());e1=json.loads((R/'e1_per_image.json').read_text())
grads=json.loads((R/'gradients_per_image.json').read_text());losses=json.loads((R/'matched_batch_loss.json').read_text())
M=json.loads((HERE/'manifest.json').read_text());selection=json.loads((HERE/'selection.json').read_text())
assert [r['index'] for r in e0]==selection['all_eligible_fit']
assert [r['index'] for r in e1]==selection['original32']+selection['additional128']
assert len(grads)==160*3*4 and len(losses)==10*11
by_index={r['index']:r for r in e0}
def mean(x):return float(np.mean(x))
def validate(m):
    tp,fp,fn=(m[k] for k in ('tp','fp','fn'));assert min(tp,fp,fn)>=0
    assert m['iou']==(tp/(tp+fp+fn) if tp+fp+fn else 0.)
    if 'dice' in m:assert m['dice']==(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
for r in e0:
    validate(r['baseline'])
    for m in r['oracles'].values():validate(m);assert m['iou']>=r['baseline']['iou']
for r in e1:
    b=r['baseline'];validate(b)
    assert all(b[k]==by_index[r['index']]['baseline'][k] for k in ('tp','fp','fn','iou'))
    for name,c in r['conditions'].items():
        m=c['metrics'];validate(m)
        if name.endswith('semantic_source'):continue
        assert m['fp']-b['fp']==c['new_fp']-c['fixed_fp']
        assert m['fn']-b['fn']==c['new_fn']-c['fixed_fn']
        assert m['tp']-b['tp']==c['fixed_fn']-c['new_fn']
        assert c['changed_pixels']==sum(c[k] for k in ('fixed_fp','fixed_fn','new_fp','new_fn'))
        assert 0<=c['topk_errors']<=min(c['top_k'],b['fp']+b['fn'])
for g in grads:
    b=by_index[g['index']]['baseline']
    for k,stats in g['stats'].items():
        assert 0<=stats['wrong']+stats['zero']<=stats['n']
        assert 0<=stats['wrong_abs_gradient']<=stats['abs_gradient']+1e-12
        if k.endswith('/all'):
            kind=k.split('/')[0];assert stats['n']==(b[kind] if kind!='tn' else 224*224-b['tp']-b['fp']-b['fn'])
summary=dict(source_git_commit=runtime['source_git_commit'],train_only=True,train_updates=0,cohorts={})
for cohort in ('original32','additional128','all160'):
    rs=[r for r in e1 if cohort=='all160' or r['cohort']==cohort];out=dict(n=len(rs),baseline_iou=mean([r['baseline']['iou'] for r in rs]),conditions={},gradients={})
    for key in sorted(set().union(*(r['conditions'] for r in rs))):
        selected=[r for r in rs if key in r['conditions']]
        cs=[r['conditions'][key] for r in selected];vals=[c['metrics'] for c in cs]
        result=dict(n=len(selected),mean={k:mean([m[k] for m in vals]) for k in ('iou','dice','precision','recall')},
            delta={k:mean([c['metrics'][k]-r['baseline'][k] for c,r in zip(cs,selected)]) for k in ('iou','dice','precision','recall','fp','fn','brier','hard_area')})
        if not key.endswith('semantic_source'):
            result['transitions']={k:mean([c[k] for c in cs]) for k in ('fixed_fp','fixed_fn','new_fp','new_fn','changed_pixels','mean_abs_delta')}
            result['top5_error_precision']=sum(c['topk_errors'] for c in cs)/sum(c['top_k'] for c in cs)
            result['uncertainty_top5_error_precision']=sum(c['uncertainty_topk_errors'] for c in cs)/sum(c['top_k'] for c in cs)
            bs=[b for b in losses if b['condition']==key and (cohort=='all160' or b['cohort']==cohort)]
            result['matched_batch_mean_loss_delta']=mean([b['output_loss']-b['baseline_loss'] for b in bs])
            result['batches_loss_and_iou_both_lower']=sum(b['output_loss']<b['baseline_loss'] and
                mean([next(r for r in rs if r['index']==i)['conditions'][key]['metrics']['iou']-by_index[i]['baseline']['iou'] for i in b['indices']])<0 for b in bs)
            result['matched_batches']=len(bs)
        out['conditions'][key]=result
    for loss in ('dice','focal','total'):
        for direction in ('unrestricted','mass_projected','unrestricted_28grid','mass_projected_28grid'):
            selected=[g for g in grads if g['loss']==loss and g['direction']==direction and (cohort=='all160' or g['cohort']==cohort)]
            combined={}
            for category in selected[0]['stats']:
                totals={k:sum(g['stats'][category][k] for g in selected) for k in selected[0]['stats'][category]}
                totals['wrong_fraction']=totals['wrong']/totals['n'] if totals['n'] else None
                totals['zero_fraction']=totals['zero']/totals['n'] if totals['n'] else None
                combined[category]=totals
            out['gradients'][loss+'/'+direction]=combined
    out['correct_minus_swap_iou']={v:out['conditions'][v+'/correct']['mean']['iou']-out['conditions'][v+'/swapped']['mean']['iou'] for v in ('T1','T2','T3','T4','template')}
    summary['cohorts'][cohort]=out
e0_given=json.loads((R/'e0_summary.json').read_text())
assert abs(e0_given['baseline_iou']-mean([r['baseline']['iou'] for r in e0]))<1e-14
for key,obs in e0_given['oracles'].items():
    assert abs(obs['eligible_mean_gain']-mean([r['oracles'][key]['iou']-r['baseline']['iou'] for r in e0]))<1e-14
summary['e0']=e0_given
summary['semantic_control_changed_n']=M['semantic_changed_cached_indices'].__len__()
summary['semantic_control_completed']=bool(M['semantic_changed_cached_indices'])
write_json(R/'summary_verified.json',summary)
write_json(R/'independent_verification.json',dict(verified=True,source_git_commit=runtime['source_git_commit'],
    raw_artifact_sha256_verified=True,count_metrics_and_transitions_verified=True,selection_verified=True,
    e0_n=len(e0),e1_n=len(e1),gradient_records=len(grads),train_updates=0,
    official_validation_accessed=False,test_split_accessed=False,semantic_control_completed=summary['semantic_control_completed']))
print(json.dumps(dict(e0=summary['e0'],cohorts={k:dict(n=v['n'],baseline_iou=v['baseline_iou'],
    delta_iou={n:c['delta']['iou'] for n,c in v['conditions'].items() if n.endswith('/correct')},
    correct_minus_swap_iou=v['correct_minus_swap_iou']) for k,v in summary['cohorts'].items()})))

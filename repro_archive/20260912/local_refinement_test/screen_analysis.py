"""Preregistered, descriptive head-crossfit gates; never authorize Test directly."""
import numpy as np
from refiner_names import VARIANTS

def group_interval(rows,values,repeats=10000):
    groups,inverse=np.unique([r['group_id'] for r in rows],return_inverse=True)
    sums=np.bincount(inverse,weights=np.asarray(values,dtype=float));counts=np.bincount(inverse)
    rng=np.random.default_rng(1219);means=[]
    for start in range(0,repeats,128):
        ix=rng.integers(len(groups),size=(min(128,repeats-start),len(groups)))
        means.extend(sums[ix].sum(1)/counts[ix].sum(1))
    return np.quantile(means,[.025,.975]).tolist()

def compare(rows,left,right):
    ds=[r['outputs'][left]['iou']-r['outputs'][right]['iou'] for r in rows]
    small=[r for r in rows if r['small_by_fit_cutoff']]
    return dict(n=len(rows),delta_iou=float(np.mean(ds)),group_ci95=group_interval(rows,ds),
        delta={k:float(np.mean([r['outputs'][left][k]-r['outputs'][right][k] for r in rows])) for k in ('dice','precision','recall','brier','fp','fn')},
        small_n=len(small),small_delta={k:float(np.mean([r['outputs'][left][k]-r['outputs'][right][k] for r in small])) for k in ('dice','recall')},
        fold_delta_iou={str(f):float(np.mean([r['outputs'][left]['iou']-r['outputs'][right]['iou'] for r in rows if r['fold']==f])) for f in range(5)})

def gate(value,thresholds):
    checks=dict(iou_minimum=value['delta_iou']>=thresholds['minimum_oof_iou_gain'],group_ci_positive=value['group_ci95'][0]>0,
        positive_folds=sum(v>0 for v in value['fold_delta_iou'].values())>=thresholds['minimum_positive_folds'],
        dice_no_drop=value['delta']['dice']>=0,precision_no_drop=value['delta']['precision']>=0,
        small_dice_no_drop=value['small_delta']['dice']>=0,small_recall_no_drop=value['small_delta']['recall']>=0,
        brier_no_increase=value['delta']['brier']<=0)
    return dict(checks=checks,passed=all(checks.values()))

def summarize(rows,manifest):
    pairs=[(v,'baseline') for v in VARIANTS]+[('fine_free','coarse_free'),('fine_mass','coarse_mass'),
        ('coarse_mass','coarse_free'),('fine_mass','fine_free')]
    comparisons={a+'-'+b:compare(rows,a,b) for a,b in pairs}
    gates={v:gate(comparisons[v+'-baseline'],manifest['gate']) for v in VARIANTS}
    fine={}
    for v,control in (('fine_free','coarse_free'),('fine_mass','coarse_mass')):
        increment=comparisons[v+'-'+control]
        fine[v]=dict(baseline_gate=gates[v]['passed'],matching_coarse_increment=increment['delta_iou']>0 and increment['group_ci95'][0]>0)
        fine[v]['passed']=all(fine[v].values())
    passed=[v for v in ('fine_free','fine_mass') if fine[v]['passed']]
    chosen=max(passed,key=lambda v:comparisons[v+'-baseline']['delta_iou']) if passed else None
    means={v:{k:float(np.mean([r['outputs'][v][k] for r in rows])) for k in ('iou','dice','precision','recall','brier')} for v in ('baseline',)+VARIANTS}
    changes={v:{k:float(np.mean([r['diagnostics'][v][k] for r in rows])) for k in ('fixed_fp','fixed_fn','new_fp','new_fn','changed_pixels','candidate_mean_abs_delta')} for v in VARIANTS}
    return dict(kind='fivefold_frozen_R2_head_crossfit_original_B_fit_only',n=len(rows),means=means,
        comparisons=comparisons,baseline_gates=gates,fine_interface_gates=fine,transitions=changes,
        recommended_interface_for_text_screen=chosen,automatic_full_training=False,automatic_test=False,
        original_R2_trained_on_these_images=True,bootstrap_replicates=10000,multiple_comparisons_corrected=False)

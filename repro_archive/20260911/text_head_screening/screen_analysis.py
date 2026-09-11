"""One final internal holdout assessment; no official Val/Test metrics."""
import numpy as np
from analysis import interval
from heads_names import VARIANTS


def group_interval(rows, differences):
    groups=sorted({r['group_id'] for r in rows})
    sums=np.array([sum(d for r,d in zip(rows,differences) if r['group_id']==g) for g in groups])
    counts=np.array([sum(r['group_id']==g for r in rows) for g in groups])
    rng=np.random.default_rng(1219); values=[]
    for start in range(0,10000,128):
        ix=rng.integers(len(groups),size=(min(128,10000-start),len(groups)))
        values.extend(sums[ix].sum(1)/counts[ix].sum(1))
    return np.quantile(values,[.025,.975]).tolist()


def compare(rows,left,right):
    differences=[r['outputs'][left]['iou']-r['outputs'][right]['iou'] for r in rows]
    return dict(n=len(rows),delta_iou=float(np.mean(differences)),
        image_ci95=interval(differences),group_ci95=group_interval(rows,differences),
        delta={k:float(np.mean([r['outputs'][left][k]-r['outputs'][right][k] for r in rows]))
            for k in ('dice','precision','recall','brier','soft_area','hard_area','fp','fn')})


def summarize(rows,small_cutoff,thresholds):
    eligible=[r for r in rows if r['eligible']]
    small=[r for r in rows if r['outputs']['baseline']['gt_area']<=small_cutoff]
    subsets={'all':rows,'eligible':eligible,'small':small,
        'rare_template_lt10':[r for r in rows if r['template_frequency']<10],
        'common_template_ge100':[r for r in rows if r['template_frequency']>=100],
        'unilateral':[r for r in rows if r['semantic_group']=='unilateral'],
        'asymmetric_bilateral':[r for r in rows if r['semantic_group']=='asymmetric_bilateral']}
    output_names=sorted(rows[0]['outputs'])
    means={name:{k:float(np.mean([r['outputs'][name][k] for r in rows])) for k in
        ('iou','dice','precision','recall','brier')} for name in output_names}
    contrasts={}
    pairs=[(v,'baseline') for v in VARIANTS]+[('T2','T1'),('T3','T1'),('T4','T2'),('T4','T3'),
        ('T3','image'),('T3','template'),('T4','image'),('T4','template')]+[(v,v+'_wrong') for v in ('T1','T3','T4')]
    for a,b in pairs:
        contrasts[a+'-'+b]={key:compare(sub,a,b) for key,sub in subsets.items() if sub}
    gates={}
    for candidate,controls in [('T3',('image','template')),('T4',('T3','image','template'))]:
        baseline=contrasts[candidate+'-baseline']
        checks=dict(eligible_iou_minimum=baseline['eligible']['delta_iou']>=thresholds['minimum_eligible_delta_iou'],
            eligible_group_ci_positive=baseline['eligible']['group_ci95'][0]>0,
            dice_no_drop=baseline['all']['delta']['dice']>=0,
            precision_no_drop=baseline['all']['delta']['precision']>=0,
            small_dice_no_drop=baseline['small']['delta']['dice']>=0,
            small_recall_no_drop=baseline['small']['delta']['recall']>=0,
            brier_no_increase=baseline['all']['delta']['brier']<=0,
            correct_text_better_than_wrong=contrasts[candidate+'-'+candidate+'_wrong']['eligible']['group_ci95'][0]>0)
        for control in controls:
            c=contrasts[candidate+'-'+control]['eligible']
            checks['increment_over_'+control]=c['delta_iou']>0 and c['group_ci95'][0]>0
        gates[candidate]=dict(checks=checks,passed=all(checks.values()))
    activation={v:dict(mean_abs_residual=float(np.mean([r['diagnostics'][v]['mean_abs_delta'] for r in eligible])),
        max_mass_error_pixels=max(r['diagnostics'][v]['absolute_mass_error'] for r in rows),
        changed_images=sum(r['diagnostics'][v]['changed_pixels']>0 for r in eligible),
        eligible_images=len(eligible)) for v in VARIANTS}
    return dict(kind='frozen_R2_Train_internal_head_holdout_not_formal_generalization',n=len(rows),
        eligible_n=len(eligible),small_cutoff_fit_gt_pixels=small_cutoff,means=means,contrasts=contrasts,
        activation=activation,screen_gates=gates,automatic_full_training=False,
        official_validation_accessed=False,test_split_accessed=False,
        bootstrap_replicates=10000,bootstrap_seed=1219,
        multiple_comparisons_corrected=False)

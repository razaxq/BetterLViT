"""Frozen R2/S1/S2 Val gates; reports all conditions without Test access."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def compare(a,b,regional_increment=False):
    for d in (a,b):
        assert d['split']=='validation' and not d['test_split_accessed']
        assert len(d['checkpoint_git_commit'])==40 and d['checkpoint_git_commit']==d['analysis_git_commit']
        assert d['epochs']==80 and d['samples']==1429 and d['threshold']==.5 and d['selection_metric']=='iou'
        assert not d['text_use_lora'] and d['boundary_loss_weight']==0
        assert d['training_recipe']['lr_schedule']=='single_cosine'
        assert d['training_recipe']['augmentation_policy']=='legacy'
    assert a['seed']==b['seed']
    assert a['experiment']==('s1_r2_pixel_aux' if regional_increment else 'r2_single_cosine')
    assert b['experiment'] in ('s1_r2_pixel_aux','s2_r2_visual_aux')
    if regional_increment:assert b['experiment']=='s2_r2_visual_aux'
    aa={r['name']:r for r in a['records']};bb={r['name']:r for r in b['records']}
    assert len(aa)==len(bb)==1429 and aa.keys()==bb.keys()
    names=sorted(aa);areas=np.array([aa[n]['label_pixels'] for n in names])
    assert np.array_equal(areas,[bb[n]['label_pixels'] for n in names])
    small=areas<=np.quantile(areas,.25)
    rng=np.random.default_rng(1219);result={}
    for key in ('iou','dice','precision','recall','brier'):
        av=np.array([aa[n][key] for n in names]);bv=np.array([bb[n][key] for n in names])
        assert np.isfinite(av).all() and np.isfinite(bv).all()
        assert abs(av.mean()-a['macro_'+key])<1e-12 and abs(bv.mean()-b['macro_'+key])<1e-12
        delta=bv-av
        boot=np.concatenate([delta[rng.integers(len(names),size=(200,len(names)))].mean(1) for _ in range(50)])
        result[key]=dict(mean=float(delta.mean()),small_mean=float(delta[small].mean()),ci95=np.quantile(boot,[.025,.975]).tolist())
    minimum=.001 if regional_increment else .003
    checks=dict(iou_minimum=result['iou']['mean']>=minimum,iou_ci_positive=result['iou']['ci95'][0]>0,
        dice_not_lower=result['dice']['mean']>=0,precision_not_lower=result['precision']['mean']>=0,
        small_dice_not_lower=result['dice']['small_mean']>=0,small_recall_not_lower=result['recall']['small_mean']>=0,
        brier_not_higher=result['brier']['mean']<=0)
    return dict(control=a['experiment'],candidate=b['experiment'],seed=a['seed'],
        control_sha=a['checkpoint_git_commit'],candidate_sha=b['checkpoint_git_commit'],
        regional_increment=regional_increment,minimum_iou_delta=minimum,deltas=result,checks=checks,
        passed=all(checks.values()),small_area_cutoff=float(np.quantile(areas,.25)),small_count=int(small.sum()),
        split='validation',test_split_accessed=False,stable_gain_proven=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--control',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--regional-increment',action='store_true');args=p.parse_args()
    r=compare(json.loads(args.control.read_text()),json.loads(args.candidate.read_text()),args.regional_increment)
    r['input_sha256']={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in (args.control,args.candidate)}
    args.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))

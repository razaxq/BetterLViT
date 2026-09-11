"""Independent CPU metrics and paired summaries, with strict >0.5 convention."""
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    temp.replace(path)


def metrics(probability, label):
    p=np.asarray(probability,dtype=np.float64).reshape(-1)
    y=np.asarray(label,dtype=bool).reshape(-1); pred=p>.5
    assert p.shape==y.shape and p.size and np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
    tp=int((pred&y).sum()); fp=int((pred&~y).sum()); fn=int((~pred&y).sum())
    return dict(iou=tp/(tp+fp+fn) if tp+fp+fn else 0.,dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,
        brier=float(np.square(p-y).mean()),soft_area=float(p.sum()),hard_area=int(pred.sum()),
        tp=tp,fp=fp,fn=fn,gt_area=int(y.sum()))


def interval(values, seed=1219):
    x=np.asarray(values,dtype=np.float64)
    if not len(x):return None
    if np.all(x==x[0]):return [float(x[0]),float(x[0])]
    rng=np.random.default_rng(seed); samples=[]
    for start in range(0,10000,128):
        ix=rng.integers(0,len(x),size=(min(128,10000-start),len(x)))
        samples.extend(x[ix].mean(1))
    return np.quantile(samples,[.025,.975]).tolist()


def summarize(rows):
    if not rows:raise ValueError('No observations')
    keys=sorted(rows[0]['conditions']); out={}
    strata={'all':lambda r:True,
        'unilateral':lambda r:r['group']=='unilateral',
        'asymmetric_bilateral':lambda r:r['group']=='asymmetric_bilateral',
        'symmetric_bilateral':lambda r:r['group']=='symmetric_bilateral',
        'unparsed':lambda r:r['group']=='unparsed',
        'train_template_frequency_lt10':lambda r:r['train_template_frequency']<10,
        'train_template_frequency_ge100':lambda r:r['train_template_frequency']>=100}
    for condition in keys:
        entries={}
        for group,predicate in list(strata.items())+[('token_changed',lambda r:r['conditions'][condition]['tokens_changed'])]:
            selected=[r for r in rows if predicate(r)]
            if not selected:entries[group]=dict(n=0);continue
            base=[r['baseline'] for r in selected]; vals=[r['conditions'][condition] for r in selected]
            delta=[v['iou']-b['iou'] for v,b in zip(vals,base)]
            means={k:float(np.mean([v[k] for v in vals])) for k in ('iou','dice','precision','recall','brier','probability_mae','changed_pixels','soft_area','hard_area')}
            entries[group]=dict(n=len(selected),tokens_changed=sum(v['tokens_changed'] for v in vals),
                mean=means,delta_iou=float(np.mean(delta)),
                delta_iou_ci95=interval(delta) if group in ('all','token_changed','unilateral','asymmetric_bilateral') else None,
                delta={k:float(np.mean([v[k]-b[k] for v,b in zip(vals,base)])) for k in ('dice','precision','recall','brier','soft_area','hard_area')})
        out[condition]=entries
    binding={}
    for scope in ('main','eppa','both'):
        selected=[r for r in rows if r['group'] in ('unilateral','asymmetric_bilateral')]
        contrasts={}
        for metric in ('iou','dice','precision','recall','brier','soft_area','hard_area'):
            differences=[r['conditions']['relation_swap/'+scope][metric]-r['conditions']['canonical/'+scope][metric] for r in selected]
            contrasts[metric]=dict(mean=float(np.mean(differences)),ci95=interval(differences)) if differences else None
        binding[scope]=dict(n=len(selected),swap_minus_semantic_canonical=contrasts)
    return dict(kind='functional_text_intervention_diagnostic_not_method_gain',threshold=.5,threshold_operator='>',
        n=len(rows),bootstrap_replicates=10000,bootstrap_seed=1219,
        baseline={k:float(np.mean([r['baseline'][k] for r in rows])) for k in ('iou','dice','precision','recall','brier')},
        conditions=out,binding_controlled_contrast=binding,automatic_architecture_pass=False,
        interpretation='Wrong-text scores against original masks measure input sensitivity only. Decisions require manual review of semantic-preserving controls and matched text-capacity probes.')

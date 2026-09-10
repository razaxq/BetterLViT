"""Descriptive paired analysis of historical exports; never loads new Test images."""
import csv
import hashlib
import json
from pathlib import Path
import re
import numpy as np

HERE=Path(__file__).resolve().parent
DOCS=HERE.parents[2]
SOURCE=DOCS/'repro_archive/20260907'
OUT=HERE/'results'
METRICS=('iou','dice','precision','recall','brier','boundary_f1_tolerance_2')


def dump(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def bootstrap(delta,repetitions=10000):
    a=np.asarray(delta,np.float64);rng=np.random.default_rng(1219)
    boot=np.concatenate([a[rng.integers(len(a),size=(min(200,repetitions-i),len(a)))].mean(1) for i in range(0,repetitions,200)])
    return dict(mean=float(a.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),n=len(a))


def counts(r):
    gt,pp=int(r['label_pixels']),int(r['prediction_pixels'])
    tp=int(round(r['recall']*gt));fp=pp-tp;fn=gt-tp
    assert min(tp,fp,fn)>=0
    values=dict(iou=tp/(gt+fp) if gt+fp else 1.,dice=2*tp/(pp+gt) if pp+gt else 1.,
                precision=tp/pp if pp else 0.,recall=tp/gt if gt else 0.)
    assert all(abs(values[k]-r[k])<1e-12 for k in values)
    return dict(tp=tp,fp=fp,fn=fn,gt=gt,pp=pp)


def decompose(c,t):
    assert c['gt']==t['gt']
    f=lambda tp,fp:tp/(c['gt']+fp) if c['gt']+fp else 1.
    fp=.5*(f(c['tp'],t['fp'])-f(c['tp'],c['fp'])+f(t['tp'],t['fp'])-f(t['tp'],c['fp']))
    fn=.5*(f(t['tp'],c['fp'])-f(c['tp'],c['fp'])+f(t['tp'],t['fp'])-f(c['tp'],t['fp']))
    assert abs(fp+fn-(f(t['tp'],t['fp'])-f(c['tp'],c['fp'])))<1e-12
    return fp,fn


def group_summary(rows):
    a=np.asarray([r['delta_iou'] for r in rows]);n=len(rows)
    return dict(n=n,iou=bootstrap(a),dice_delta=float(np.mean([r['delta_dice'] for r in rows])),
                precision_delta=float(np.mean([r['delta_precision'] for r in rows])),
                recall_delta=float(np.mean([r['delta_recall'] for r in rows])),
                brier_delta=float(np.mean([r['delta_brier'] for r in rows])),
                fp_delta_mean=float(np.mean([r['c8_fp']-r['c4_fp'] for r in rows])),
                fn_delta_mean=float(np.mean([r['c8_fn']-r['c4_fn'] for r in rows])),
                fp_iou_contribution=float(np.mean([r['fp_iou_contribution'] for r in rows])),
                fn_iou_contribution=float(np.mean([r['fn_iou_contribution'] for r in rows])),
                wins=int((a>1e-12).sum()),losses=int((a < -1e-12).sum()),ties=int((np.abs(a)<=1e-12).sum()))


def analyze(split):
    folder=SOURCE/('race_pe_v2_test_20260907' if split=='test' else 'race_pe_v2_results_20260907')
    suffix='test' if split=='test' else 'validation'
    files={a:folder/(a+'_'+suffix+'.json') for a in ('c4','c8','p10')}
    data={a:json.loads(p.read_text()) for a,p in files.items()}
    tables={a:{r['name']:r for r in d['records']} for a,d in data.items()}
    assert set(tables['c4'])==set(tables['c8'])==set(tables['p10'])
    for a,d in data.items():
        assert d['threshold']==.5 and d['checkpoint_best_epoch']==80 and d['seed']==1219
        for k in METRICS:assert abs(np.mean([r[k] for r in d['records']])-d['macro_'+k])<1e-12
    rows=[]
    for name in sorted(tables['c4']):
        c,t,p=(tables[a][name] for a in ('c4','c8','p10'))
        assert c['label_pixels']==t['label_pixels']==p['label_pixels']
        cc,tc=counts(c),counts(t);counts(p)
        fp,fn=decompose(cc,tc)
        row=dict(name=name,gt=cc['gt'],**{'c4_'+k:v for k,v in cc.items()},**{'c8_'+k:v for k,v in tc.items()},
                 **{'delta_'+k:t[k]-c[k] for k in METRICS},fp_iou_contribution=fp,fn_iou_contribution=fn,
                 p10_minus_c8_iou=p['iou']-t['iou'],p10_minus_c8_precision=p['precision']-t['precision'],
                 c4_iou=c['iou'],c8_iou=t['iou'])
        rows.append(row)
    summary=group_summary(rows)
    summary.update(split=split,means={a:{k:d['macro_'+k] for k in METRICS} for a,d in data.items()},
                   pairs={k:bootstrap([r['delta_'+k] for r in rows]) for k in METRICS},
                   count_means={a:{k:float(np.mean([r[a+'_'+k] for r in rows])) for k in ('tp','fp','fn','pp','gt')} for a in ('c4','c8')},
                   input_files={a:dict(path=p.relative_to(DOCS).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                   checkpoint_sha=data[a]['checkpoint_git_commit']) for a,p in files.items()})
    by_size=sorted(rows,key=lambda r:(r['gt'],r['name']))
    summary['size_quartiles']=[dict(quartile=i+1,gt_min=min(r['gt'] for r in group),gt_max=max(r['gt'] for r in group),
        **group_summary(group)) for i,group in enumerate([list(g) for g in np.array_split(np.asarray(by_size,dtype=object),4)])]
    groups={}
    for name,predicate in [('over_area',lambda r:r['c4_pp']>r['gt']),('under_area',lambda r:r['c4_pp']<r['gt']),('equal_area',lambda r:r['c4_pp']==r['gt'])]:
        group=[r for r in rows if predicate(r)]
        if group:groups[name]=group_summary(group)
    summary['baseline_area_groups']=groups
    summary['filename_groups']={}
    for name,predicate in [('sub_S',lambda r:r['name'].startswith('sub-S')),('covid',lambda r:r['name'].startswith('covid_'))]:
        group=[r for r in rows if predicate(r)]
        if group:summary['filename_groups'][name]=group_summary(group)
    delta=np.asarray([r['delta_iou'] for r in rows]);ordered=np.sort(delta)
    summary['delta_distribution']=dict(quantiles=dict(zip(('min','p05','p25','median','p75','p95','max'),np.quantile(delta,[0,.05,.25,.5,.75,.95,1]).tolist())),
        trimmed_means={str(q):float(ordered[int(len(rows)*q):-int(len(rows)*q)].mean()) for q in (.01,.05)},
        positive_sum=float(delta[delta>0].sum()),negative_sum=float(delta[delta<0].sum()),
        top10_wins_sum=float(ordered[-10:].sum()),top10_losses_sum=float(ordered[:10].sum()))
    summary['routing_difference']={k:bootstrap([r['p10_minus_c8_'+k] for r in rows]) for k in ('iou','precision')}
    # Cluster sensitivity is restricted to filenames with an observed subject ID.
    clusters={}
    for row in rows:
        match=re.match(r'sub-(S\d+)_',row['name'])
        if match:clusters.setdefault(match.group(1),[]).append(row['delta_iou'])
    if clusters:
        sums=np.asarray([sum(v) for v in clusters.values()]);ns=np.asarray([len(v) for v in clusters.values()]);rng=np.random.default_rng(1219)
        vals=[]
        for _ in range(50):
            ix=rng.integers(0,len(ns),(200,len(ns)));vals.extend((sums[ix].sum(1)/ns[ix].sum(1)).tolist())
        summary['subject_cluster_sensitivity']=dict(known_subjects=len(clusters),known_images=int(ns.sum()),
            maximum_images_per_subject=int(ns.max()),mean=float(sums.sum()/ns.sum()),ci95=np.quantile(vals,[.025,.975]).tolist(),
            scope='Only sub-S filename subjects; no invented IDs for other images.')
    with (OUT/(split+'_paired.csv')).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    dump(OUT/(split+'_analysis.json'),summary)
    dump(OUT/(split+'_extreme_cases.json'),dict(largest_gains=sorted(rows,key=lambda r:r['delta_iou'])[-10:],largest_losses=sorted(rows,key=lambda r:r['delta_iou'])[:10]))
    return summary


def main():
    OUT.mkdir(exist_ok=True)
    results={s:analyze(s) for s in ('validation','test')}
    print(json.dumps({s:{k:v for k,v in d.items() if k in ('n','iou','count_means','fp_iou_contribution','fn_iou_contribution','wins','losses','ties','size_quartiles','subject_cluster_sensitivity')} for s,d in results.items()},indent=2))


if __name__=='__main__':main()

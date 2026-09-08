"""Descriptive paired error strata from completed, saved Val results only."""
import argparse
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from prepare_recipe import HERE, ROOT

p=argparse.ArgumentParser()
p.add_argument('--label',choices=('r1','r2'),required=True)
label=p.parse_args().label
control=json.loads((HERE.parent/'visual_prior_execution/c4_validation.json').read_text())
candidate=json.loads((ROOT/f'{label}_validation.json').read_text())
snap=json.loads((ROOT/f'{label}_final_snapshot.json').read_text(encoding='utf-8'))
forecast=json.loads((ROOT/f'{label}_forecast.json').read_text())
a={r['name']:r for r in control['records']}
b={r['name']:r for r in candidate['records']}
assert len(a)==len(b)==1429 and a.keys()==b.keys()
names=sorted(a)
areas=np.array([a[n]['label_pixels'] for n in names])
assert all(a[n]['label_pixels']==b[n]['label_pixels'] for n in names)
cuts=np.quantile(areas,[.25,.5,.75])
groups=[]
for index in range(4):
    mask=np.ones(len(names),dtype=bool)
    if index:mask &= areas>cuts[index-1]
    if index<3:mask &= areas<=cuts[index]
    selected=[n for n,yes in zip(names,mask) if yes]
    groups.append(dict(gt_area_quartile=index+1,samples=len(selected),
        metrics={key:dict(control=float(np.mean([a[n][key] for n in selected])),
            candidate=float(np.mean([b[n][key] for n in selected])),
            delta=float(np.mean([b[n][key]-a[n][key] for n in selected])))
            for key in ('iou','dice','precision','recall')}))


def counts(records):
    totals=dict(tp=0,fp=0,fn=0,predicted_pixels=0,label_pixels=0)
    for row in records.values():
        tp=round(row['recall']*row['label_pixels'])
        assert abs(tp-row['recall']*row['label_pixels'])<1e-7
        if row['prediction_pixels']:assert abs(tp/row['prediction_pixels']-row['precision'])<1e-12
        for key,value in dict(tp=tp,fp=row['prediction_pixels']-tp,fn=row['label_pixels']-tp,
            predicted_pixels=row['prediction_pixels'],label_pixels=row['label_pixels']).items():totals[key]+=value
    return totals


at,bt=counts(a),counts(b)
runtime=snap['files']['runtime.json']
local=lambda t:datetime.fromtimestamp(t,ZoneInfo('Australia/Sydney')).isoformat()
value=dict(label=label,split='validation',test_split_accessed=False,source_git_commit=candidate['checkpoint_git_commit'],
    size_quartile_cutoffs=cuts.tolist(),size_quartiles=groups,
    pooled_pixel_counts_diagnostic_only=dict(control=at,candidate=bt,deltas={k:bt[k]-at[k] for k in at}),
    training_ended_sydney=local(runtime['training_ended_unix']),
    validation_ended_sydney=local(runtime['validation_ended_unix']),final_inspected_sydney=local(snap['inspected_unix']),
    prediction_error_seconds=runtime['training_ended_unix']-forecast['predicted_training_end_unix'],
    seconds_after_training=snap['seconds_after_training'],
    interpretation='Descriptive error strata; does not replace the registered overall IoU gate or prove causal mechanism.')
(HERE/(label+'_results')/'error_analysis.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
print(json.dumps(value))

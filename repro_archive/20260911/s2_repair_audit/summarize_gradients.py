"""Summarize existing Train-only feature-gradient observations; no remote calls."""
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
root=HERE.parents[1]/'20260910/visual_aux_execution'
result={}


def summarize(values):
    return dict(observations=len(values),negative_cosine=sum(x['cosine']<0 for x in values),
                ratio_min=min(x['ratio'] for x in values),ratio_max=max(x['ratio'] for x in values),
                cosine_min=min(x['cosine'] for x in values),cosine_max=max(x['cosine'] for x in values),
                mean_cosine=float(np.mean([x['cosine'] for x in values])))


for label in ('s1','s2'):
    items=[json.loads((root/(label+'_results/diagnostics')/f'epoch_{e:03d}.json').read_text()) for e in (20,40,60,80)]
    result[label]={}
    for term in items[0]['batches'][0]['gradients']:
        if term=='main':continue
        values=[x for d in items for b in d['batches'] for x in b['gradients'][term]]
        result[label][term]=summarize(values)
        result[label][term]['by_epoch']={str(d['epoch']):summarize(
            [x for b in d['batches'] for x in b['gradients'][term]]) for d in items}
        result[label][term]['by_feature_tap']={name:summarize(
            [b['gradients'][term][i] for d in items for b in d['batches']])
            for i,name in enumerate(('inc','down1','down2','down3'))}
        for grouping in ('by_epoch','by_feature_tap'):
            assert sum(x['observations'] for x in result[label][term][grouping].values())==len(values)
            assert sum(x['negative_cosine'] for x in result[label][term][grouping].values())==result[label][term]['negative_cosine']
(HERE/'gradient_snapshot_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))

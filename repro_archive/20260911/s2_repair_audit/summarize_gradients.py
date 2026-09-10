"""Summarize existing Train-only feature-gradient observations; no remote calls."""
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
root=HERE.parents[1]/'20260910/visual_aux_execution'
result={}
for label in ('s1','s2'):
    items=[json.loads((root/(label+'_results/diagnostics')/f'epoch_{e:03d}.json').read_text()) for e in (20,40,60,80)]
    result[label]={}
    for term in items[0]['batches'][0]['gradients']:
        if term=='main':continue
        values=[x for d in items for b in d['batches'] for x in b['gradients'][term]]
        result[label][term]=dict(observations=len(values),negative_cosine=sum(x['cosine']<0 for x in values),ratio_min=min(x['ratio'] for x in values),ratio_max=max(x['ratio'] for x in values),mean_cosine=float(np.mean([x['cosine'] for x in values])))
(HERE/'gradient_snapshot_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))

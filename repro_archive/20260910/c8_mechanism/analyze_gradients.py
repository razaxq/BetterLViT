"""Aggregate four fixed diagnostic batches without treating them as replications."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
d=json.loads((root/'evidence/c8_gradient_probe.json').read_text())
h=json.loads((root/'evidence/c8_history.json').read_text())
assert len(d['batches'])==4 and len(set(d['samples']))==32
assert d['state_sha256_before']==d['state_sha256_after']==h['state_sha256']
assert d['script_sha256']==hashlib.sha256((root/'remote_probe.py').read_bytes()).hexdigest()
assert all(b['inference_max_abs_diff']==0 and b['slot_head_gradient_norm']>0 for b in d['batches'])
rows=[]
for term in ('pixel','presence','occupancy','visual_aux','text','count'):
    for i,layer in enumerate(('inc','down1','down2','down3')):
        values=[b['gradients'][term][i] for b in d['batches']]
        connected=all(x['connected'] for x in values)
        assert connected==(term not in ('text','count'))
        row=dict(term=term,layer=layer,connected=connected,
            ratio_mean=float(np.mean([x['ratio'] for x in values])),
            ratio_min=min(x['ratio'] for x in values),ratio_max=max(x['ratio'] for x in values),
            cosine_mean=float(np.mean([x['cosine'] for x in values])) if connected else None,
            cosine_min=min(x['cosine'] for x in values) if connected else None,
            cosine_max=max(x['cosine'] for x in values) if connected else None)
        rows.append(row)
out=dict(scope=d['scope'],aggregation='Arithmetic means of four batchwise feature-gradient statistics; not parameter gradients or training-time integrals.',
         assertions_passed=True,layers=rows,
         weighted_loss_mean={k:float(np.mean([b['weighted_losses'][k] for b in d['batches']]))
                             for k in d['batches'][0]['weighted_losses']})
(root/'results/gradient_analysis.json').write_text(json.dumps(out,indent=2)+'\n')
with (root/'results/gradient_by_layer.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print('Verified state hashes, source hash, 32 exact outputs, and gradient connectivity.')

"""Validate and archive the saved final snapshot without another remote check."""
import hashlib
import json
import math
from pathlib import Path

root = Path('D:/BetterLViT/outputs/visual_prior_20260908')
out = Path(__file__).resolve().parent / 'probe_results'
out.mkdir(exist_ok=True)
s = json.loads((root/'probe_final_snapshot.json').read_text())
status = s['files']['status.json']
assert status['stage'] == 'complete' and not status['test_split_accessed']
assert not s['tracked_changes']
assert status['source_git_commit'] == s['source_git_commit'] == '5d7f4eeca92462de485d972b4cd7eebb381a07b5'
a, b = (s['files'][kind+'_validation.json'] for kind in ('cxformer','dinov2'))
for kind, value in [('cxformer', a), ('dinov2', b)]:
    assert len(value['records']) == 1429 and value['split'] == 'validation'
    assert value['threshold'] == .5 and not value['test_split_accessed']
    assert value['source_git_commit'] == s['source_git_commit']
    assert len(s['files'][kind+'_history.json']) == 20
    for metric in ('iou', 'dice'):
        assert all(math.isfinite(x[metric]) for x in value['records'])
        assert abs(sum(x[metric] for x in value['records'])/1429-value['macro_'+metric]) < 1e-12
left, right = (status['encoders'][kind] for kind in ('cxformer','dinov2'))
assert left['initial_head_sha256'] == right['initial_head_sha256']
for split in ('Train_Folder', 'Val_Folder'):
    for key in ('samples', 'name_sha256', 'mask_sha256', 'non_gray_images'):
        assert left['extraction'][split][key] == right['extraction'][split][key]
ar = {x['image']:x for x in a['records']}
br = {x['image']:x for x in b['records']}
assert len(ar) == len(br) == 1429 and ar.keys() == br.keys()
assert all(ar[k]['gt_area'] == br[k]['gt_area'] for k in ar)
assert all(0 <= x <= 1800 for x in s['seconds_after_each_probe'].values())
for name, value in s['files'].items():
    (out/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
summary = {'diagnostic_only':True, 'test_split_accessed':False,
    'source_git_commit':s['source_git_commit'], 'inspections_completed':2,
    'inspected_unix':s['inspected_unix'],
    'seconds_after_each_probe':s['seconds_after_each_probe'],
    'paired_contract_verified':True, 'records':1429,
    'cxformer':{k:a[k] for k in ('macro_iou','macro_dice','best_epoch')},
    'dinov2':{k:b[k] for k in ('macro_iou','macro_dice','best_epoch')},
    'cxformer_minus_dinov2':{k:a[k]-b[k] for k in ('macro_iou','macro_dice')},
    'advance_to_p12':True,
    'limitation':'Feature readout only; no full BetterLViT gain, cross-seed stability, or final Test claim.'}
(out/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary))

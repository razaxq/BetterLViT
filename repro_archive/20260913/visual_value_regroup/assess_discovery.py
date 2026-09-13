"""Apply the preregistered discovery gate after both frozen runs are complete."""
import json
from remote_ops import HERE,read,save
for label in ('m1','m2'):
    for name in ('independent_verification.json','hf_upload_verified.json','download_xet_verified.json','github_archive_verified.json'):
        assert read(HERE/(label+'_results')/name)['verified']
    assert not read(HERE/(label+'_results/independent_verification.json'))['threshold_reconciliation_needed'], 'Reconcile selection/export before interpretation'
r2=read(HERE/'m2_results/paired_comparison.json')
m1=read(HERE/'m2_results/m2_vs_m1.json')
gates=dict(m2_vs_r2_iou_at_least_point003=r2['deltas']['iou']['mean']>=.003,
    m2_vs_r2_iou_grouped_ci_lower_positive=r2['deltas']['iou']['ci95'][0]>0,
    m2_vs_m1_iou_positive=m1['deltas']['iou']['mean']>0,
    m2_vs_m1_iou_grouped_ci_lower_positive=m1['deltas']['iou']['ci95'][0]>0,
    m2_vs_r2_dice_nonregression=r2['deltas']['dice']['mean']>=0,
    m2_vs_m1_dice_nonregression=m1['deltas']['dice']['mean']>=0,
    m2_vs_r2_small_iou_nonregression=r2['deltas']['iou']['small_mean']>=0,
    m2_vs_m1_small_iou_nonregression=m1['deltas']['iou']['small_mean']>=0)
result=dict(discovery_gate_passed=all(gates.values()),gates=gates,seed=1219,
    second_innovation_confirmed=False,test_split_accessed=False,
    m2_vs_r2=r2,m2_vs_m1=m1,
    next_step='Matched additional seeds, mechanism ablations and prior-art differentiation' if all(gates.values()) else
              'Close this version; preserve negative results; do not retune the registered gate')
save('discovery_decision.json',result)
rows=['# Visual-value regrouping discovery result','',
      'All results below are single-seed validation results. The second innovation is not yet established.','',
      '| Comparison | IoU delta (pp) | Dice delta (pp) | Small-mask IoU delta (pp) |',
      '|---|---:|---:|---:|']
for name,c in [('M2 minus R2',r2),('M2 minus M1',m1)]:
    rows.append(f"| {name} | {c['deltas']['iou']['mean']*100:+.4f} | {c['deltas']['dice']['mean']*100:+.4f} | {c['deltas']['iou']['small_mean']*100:+.4f} |")
rows.extend(['','Discovery gate passed: '+str(result['discovery_gate_passed']), '',result['next_step'], '',
    'Known prior art (OCR and RecLMIS) prevents treating prototype aggregation or language conditioning alone as novelty.'])
(HERE/'COMPLETE_REPORT.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('m2_vs_r2','m2_vs_m1')},indent=2))

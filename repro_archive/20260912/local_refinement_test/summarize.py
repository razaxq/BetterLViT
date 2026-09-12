"""Verify exported pixel counts and report every fixed head, without Test selection."""
import json
from pathlib import Path
import numpy as np
from analysis import digest,write_json
from screen_analysis import group_interval
from refiner_names import VARIANTS
HERE=Path(__file__).resolve().parent;DOCS=HERE.parents[2]
METRICS=('iou','dice','precision','recall','brier','fp','fn')
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def run():
    a=read(HERE/'authorization.json');d=read(HERE/'launch.json');rt=read(HERE/'results/runtime.json')
    assert rt['phase']=='complete' and rt['evaluation_source_git_commit']==d['evaluation_source_git_commit']
    for name,meta in rt['artifacts'].items():
        path=HERE/'results'/name;assert path.stat().st_size==meta['bytes'] and digest(path)==meta['sha256']
    proof=read(HERE/'results/provenance.json')
    assert proof['evaluation_source_git_commit']==d['evaluation_source_git_commit']
    assert proof['f_source_git_commit']==a['f_source_git_commit'] and proof['head_states']==a['head_state_sha256']
    assert proof['all_weights_unchanged'] and proof['no_gradients'] and proof['historical_baseline_exact_samples']==2113
    assert proof['authorization_sha256']==digest(HERE/'authorization.json')==d['deployed_sha256']['authorization.json']
    value=read(HERE/'results/records.json');rows=value['records']
    assert value['test_split_accessed'] and value['threshold']==.5 and value['threshold_operator']=='>'
    assert len(rows)==len({r['name'] for r in rows})==2113
    historical=DOCS/a['historical_test_relative'];assert digest(historical)==a['historical_test_sha256']
    old={r['name']:r for r in read(historical)['records']};assert {r['name'] for r in rows}==set(old)
    keys=['baseline']+[f'{f}/{v}' for f in range(5) for v in VARIANTS]
    for row in rows:
        assert set(row['outputs'])==set(keys)
        b=row['outputs']['baseline']
        for metric in ('iou','dice','precision','recall'):assert b[metric]==old[row['name']][metric]
        for key,m in row['outputs'].items():
            tp,fp,fn=m['tp'],m['fp'],m['fn'];assert all(isinstance(x,int) and x>=0 for x in (tp,fp,fn))
            assert tp+fp+fn<=224*224 and tp+fn==m['gt_area']==b['gt_area'] and tp+fp==m['hard_area']
            expected=dict(iou=tp/(tp+fp+fn) if tp+fp+fn else 0.,dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.)
            for k,x in expected.items():assert x==m[k]
            assert np.isfinite(list(m.values())).all() and 0<=m['brier']<=1
            if key=='baseline':continue
            x=row['diagnostics'][key]
            assert m['fp']==b['fp']-x['fixed_fp']+x['new_fp']
            assert m['fn']==b['fn']-x['fixed_fn']+x['new_fn']
            assert x['changed_pixels']==sum(x[k] for k in ('fixed_fp','fixed_fn','new_fp','new_fn'))<=1024
            assert x['noncandidate_probabilities_exact']
            if key.endswith('mass'):assert x['soft_mass_absolute_error']<=.02
    arrays={key:{m:np.array([r['outputs'][key][m] for r in rows]) for m in METRICS} for key in keys}
    heads={key:{m:float(x.mean()) for m,x in values.items()} for key,values in arrays.items()}
    variants={};averages={}
    for variant in VARIANTS:
        averages[variant]={m:np.mean([arrays[f'{f}/{variant}'][m] for f in range(5)],axis=0) for m in METRICS}
        delta=averages[variant]['iou']-arrays['baseline']['iou']
        fold_delta=[heads[f'{f}/{variant}']['iou']-heads['baseline']['iou'] for f in range(5)]
        variants[variant]=dict(mean={m:float(x.mean()) for m,x in averages[variant].items()},
            delta={m:float((x-arrays['baseline'][m]).mean()) for m,x in averages[variant].items()},
            group_iou_ci95=group_interval(rows,delta),fold_delta_iou=fold_delta,positive_models=sum(x>0 for x in fold_delta))
    contrasts={}
    for left,right in [('fine_mass','coarse_mass'),('fine_free','coarse_free')]:
        delta=averages[left]['iou']-averages[right]['iou']
        contrasts[left+'-'+right]=dict(delta_iou=float(delta.mean()),group_iou_ci95=group_interval(rows,delta))
    summary=dict(verified=True,split='test',samples=2113,aggregation='per-image macro; then equal mean of five fixed fold-trained heads; no ensemble',
        baseline=heads['baseline'],variants=variants,heads=heads,contrasts=contrasts,
        evaluation_source_git_commit=d['evaluation_source_git_commit'],f_source_git_commit=a['f_source_git_commit'],
        threshold=.5,threshold_operator='>',primary_variant_before_test='fine_mass',
        group_count=len({r['group_id'] for r in rows}),bootstrap_replicates=10000,bootstrap_seed=1219,
        multiple_comparisons_corrected=False,independent_training_seeds=False,no_new_training=True,no_test_selection=True,
        original_f_screen_passed=False,historical_baseline_exact_samples=2113)
    write_json(HERE/'summary.json',summary)
    lines=['# F：固定模型的 Test 补测结果','',
        '用户明确要求的补测；2113张Test，R2 seed1219 Best67，阈值严格>0.5。四组各五个固定末步头完整评估。表中为各头逐图macro指标的等权平均，不是预测集成，也不是五个独立随机种子。原Train内筛选失败结论保留。','',
        '| 方案 | IoU % | Dice % | Precision % | Recall % | ΔIoU pp | 正向模型 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    b=heads['baseline'];lines.append(f"| R2 | {b['iou']*100:.6f} | {b['dice']*100:.6f} | {b['precision']*100:.6f} | {b['recall']*100:.6f} | — | — |")
    for v,x in variants.items():
        m=x['mean'];lines.append(f"| {v} | {m['iou']*100:.6f} | {m['dice']*100:.6f} | {m['precision']*100:.6f} | {m['recall']*100:.6f} | {x['delta']['iou']*100:+.6f} | {x['positive_models']}/5 |")
    lines+=['','## 全部模型头','', '| 折 | 方案 | IoU % | Dice % | ΔIoU pp |','|---|---|---:|---:|---:|']
    for f in range(5):
        for v in VARIANTS:
            m=heads[f'{f}/{v}'];lines.append(f"| {f} | {v} | {m['iou']*100:.6f} | {m['dice']*100:.6f} | {(m['iou']-b['iou'])*100:+.6f} |")
    lines+=['','## 配对不确定性及解释','']
    for v,x in variants.items():
        ci=x['group_iou_ci95'];lines.append(f"- {v}−R2：IoU {x['delta']['iou']*100:+.6f} pp；group 95% CI [{ci[0]*100:+.6f}, {ci[1]*100:+.6f}] pp；平均FP/FN变化 {x['delta']['fp']:+.4f}/{x['delta']['fn']:+.4f} 像素/图。")
    for pair,x in contrasts.items():
        ci=x['group_iou_ci95'];lines.append(f"- {pair}：IoU {x['delta_iou']*100:+.6f} pp；group 95% CI [{ci[0]*100:+.6f}, {ci[1]*100:+.6f}] pp。")
    lines+=['',f"按文件名可识别患者/其余独立文件组成{summary['group_count']}组。先按图平均五头增量，再做10000次组bootstrap；该区间只描述这一组已训练模型在Test样本上的差异，未做多重比较校正，不能代替多个独立训练种子。",'',
        'R2的2113张逐图IoU/Dice/Precision/Recall及GT/预测面积与历史输出精确一致。20头文件/state哈希、权重前后不变、无梯度、候选外不变、保量误差，以及本地逐图计数和错误转移复算均通过。','',
        '未根据Test挑折、改阈值或重训；本次没有新增文字输入，不构成文字创新成立的证据。', '',
        f"评估源 `{d['evaluation_source_git_commit']}`；模型头训练源 `{a['f_source_git_commit']}`。"]
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:summary[k] for k in ('verified','baseline','variants','contrasts','group_count')},ensure_ascii=False))
if __name__=='__main__':run()

"""Summarize both frozen Val screens and reconstruct integer confusion counts."""
import csv
import json
import numpy as np
from remote_ops import HERE,save


def main():
    paths={'r2':HERE.parents[1]/'20260908/recipe_execution/r2_results/validation.json',
           's1':HERE/'s1_results/validation.json','s2':HERE/'s2_results/validation.json'}
    data={k:json.loads(p.read_text()) for k,p in paths.items()}
    counts={};rows=[];max_error=0.0
    for label,d in data.items():
        assert d['split']=='validation' and not d['test_split_accessed'] and d['seed']==1219
        counts[label]={}
        for r in d['records']:
            gt=r['label_pixels'];pred=r['prediction_pixels'];tp=round(r['recall']*gt)
            assert 0<=tp<=min(gt,pred)
            fp=pred-tp;fn=gt-tp
            reproduced={'iou':tp/(tp+fp+fn) if tp+fp+fn else 0,
                'dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
                'precision':tp/pred if pred else 0,'recall':tp/gt if gt else 0}
            for key,value in reproduced.items():
                max_error=max(max_error,abs(value-r[key]));assert abs(value-r[key])<1e-12
            row=dict(model=label,name=r['name'],tp=tp,fp=fp,fn=fn,label_pixels=gt,prediction_pixels=pred)
            counts[label][r['name']]=row;rows.append(row)
    assert counts['r2'].keys()==counts['s1'].keys()==counts['s2'].keys()
    names=sorted(counts['r2']);cutoff=float(np.quantile([counts['r2'][n]['label_pixels'] for n in names],.25))
    small=[n for n in names if counts['r2'][n]['label_pixels']<=cutoff]
    paired={}
    for a,b in [('r2','s1'),('r2','s2'),('s1','s2')]:
        for n in names:assert counts[a][n]['label_pixels']==counts[b][n]['label_pixels']
        paired[a+'_vs_'+b]={scope:{key:float(np.mean([counts[b][n][key]-counts[a][n][key] for n in group]))
                         for key in ('tp','fp','fn','prediction_pixels')} for scope,group in [('all',names),('small',small)]}
    gates={k:json.loads((HERE/p).read_text()) for k,p in {
        'r2_vs_s1':'s1_results/r2_vs_s1.json','r2_vs_s2':'s2_results/r2_vs_s2.json','s1_vs_s2':'s2_results/s1_vs_s2.json'}.items()}
    assert not any(g['passed'] for g in gates.values())
    gradient_ranges={}
    for label in ('s1','s2'):
        checks=json.loads((HERE/(label+'_results/independent_verification.json')).read_text());assert checks['verified']
        gradient_ranges[label]={}
        for term in checks['diagnostics'][0]['gradients']:
            values=[v for x in checks['diagnostics'] for v in x['gradients'][term]['ratio']]
            cosines=[v for x in checks['diagnostics'] for v in x['gradients'][term]['cosine']]
            gradient_ranges[label][term]=dict(ratio_min=min(values),ratio_max=max(values),cosine_min=min(cosines),cosine_max=max(cosines))
    result=dict(split='validation',test_split_accessed=False,seed=1219,samples=1429,small_count=len(small),small_area_cutoff=cutoff,
        integer_counts_verified=True,reconstructed_metric_max_abs_error=max_error,paired_mean_count_changes=paired,
        train_gradient_ranges=gradient_ranges,gradient_scope='Ranges over 4 epochs x 4 stages, each value a four-batch mean on 32 fixed Train images in eval mode',
        decisions={k:dict(passed=g['passed'],failed=[c for c,v in g['checks'].items() if not v],iou=g['deltas']['iou']) for k,g in gates.items()},
        phase='completed_validation_screen',continue_matched_seeds=False,proceed_to_test=False,extend_to_150=False,
        stable_gain_proven=False,second_innovation_established=False)
    save('ablation_summary.json',result)
    with (HERE/'val_confusion_counts.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    out=['# R2视觉辅助监督消融：阶段结论','',
        '两组均完成80轮，但均未通过预注册R2筛选门；S2也未通过相对S1的区域增量门。本轮到此停止，不进入多种子、Test或150轮延长。当前没有可确立的第二创新成果。', '',
        '| 模型 | Val IoU | Val Dice | Precision | Recall | Best |','|---|---:|---:|---:|---:|---:|']
    for label,d in data.items():
        out.append('| '+label.upper()+' | '+' | '.join(f"{100*d['macro_'+k]:.4f}%" for k in ('iou','dice','precision','recall'))+f" | {d['checkpoint_best_epoch']} |")
    out+=['','全部为1429张Val、seed1219、阈值0.5、Val IoU选Best。不是Test成绩；R2/S1/S2共享初始化和随机流的启动前核验已通过。','','| 比较 | IoU差值（百分点） | 95%图像配对区间 | 通过 |','|---|---:|---|---|']
    for label,g in gates.items():
        d=g['deltas']['iou'];out.append(f"| {label} | {100*d['mean']:+.4f} | [{100*d['ci95'][0]:+.4f}, {100*d['ci95'][1]:+.4f}] | 否 |")
    out+=['','## 从对照中得到的证据','',
        '像素监督S1有小幅正向数值，但未达+0.3个百分点且区间跨0。S2加入存在/占比后没有额外IoU收益；其precision提高、recall下降，最小GT总面积组表现更差。这支持“当前区域监督组合改变了误检与漏检的取舍”，不能解释成已提高分割能力，也不能仅凭一个种子断言辅助监督普遍无效。','']
    g=gates['s1_vs_s2'];c=paired['s1_vs_s2']['all']
    out.append(f"S2对S1：precision {100*g['deltas']['precision']['mean']:+.4f}、recall {100*g['deltas']['recall']['mean']:+.4f}个百分点；每图FP {c['fp']:+.2f}像素、FN {c['fn']:+.2f}像素。最小面积组358张（GT≤2075像素），IoU {100*g['deltas']['iou']['small_mean']:+.4f}、Dice {100*g['deltas']['dice']['small_mean']:+.4f}、recall {100*g['deltas']['recall']['small_mean']:+.4f}个百分点。该分组是总面积，不等于独立连通病灶检出率。")
    out+=['','## 诊断边界','',
        '80轮历史、学习率、Best选择和四次Train诊断状态均核验通过。S1/S2都已产生视觉辅助梯度，不能把失败归因于分支完全未生效。梯度记录是20/40/60/80轮各32张Train的eval快照，不能证明训练全程机制，也不能将梯度范数换算为IoU贡献。存在/占比同时加入，无法单独识别谁造成变化。','',
        '本次结论限于冻结系数、R2配方和seed1219。模型/结果留存供复查；不使用当前Val负结果临时调权重、改阈值或改门槛。后续若另开研究，应提出新机制与独立可证伪预检；不能把这两个配置包装为已成功创新。','',
        '复算代码：`summarize_ablation.py`；4287行计数在`val_confusion_counts.csv`，逐模型结果和全部门条件在`s1_results/`与`s2_results/`。云端备份及预约结束记录见执行README。']
    (HERE/'SUMMARY.md').write_text('\n'.join(out)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('train_gradient_ranges','decisions')},ensure_ascii=False))


if __name__=='__main__':main()

"""Readable full-metric report derived from the already registered aggregate."""
import json
import numpy as np
from remote_ops import HERE,read,save

def main():
    summary=read(HERE/'replication_summary.json');assert summary['verified']
    rows=summary['all_three_seed_results'];assert [r['seed'] for r in rows]==[1219,2027,3407]
    metrics=('iou','dice','precision','recall','brier');stats={}
    for scope,subset in (('new_seeds',rows[1:]),('all_three_including_discovery',rows)):
        stats[scope]={}
        for metric in metrics:
            values={side:np.array([r[side][metric] for r in subset]) for side in ('baseline','candidate')}
            values['delta']=values['candidate']-values['baseline']
            for d,r in zip(values['delta'],subset):assert abs(d-r['deltas'][metric]['mean'])<1e-12
            stats[scope][metric]={side:dict(mean=float(v.mean()),sample_sd=float(v.std(ddof=1))) for side,v in values.items()}
            stats[scope][metric]['small_mean_delta']=float(np.mean([r['deltas'][metric]['small_mean'] for r in subset]))
    assert abs(stats['new_seeds']['iou']['delta']['mean']-summary['new_seed_mean_iou_delta'])<1e-12
    assert abs(stats['all_three_including_discovery']['iou']['delta']['mean']-summary['all_three_seed_mean_iou_delta'])<1e-12
    save('full_metric_summary.json',dict(verified=True,stats=stats,primary_seed_scope=[2027,3407],discovery_seed=1219,
        criterion_unchanged=True,passed=summary['passed'],selected_common_recipe=summary['selected_common_recipe'],test_split_accessed=False))
    lines=['# RS1 独立种子复验完成报告','',
        '2026-09-13。两个新种子均从头训练80轮并完成1429张Val，保持原固定系数和单次余弦配方。逐图macro、严格>0.5，Val IoU选Best。本批不新增Test访问。','',
        '**结论：两个新种子的IoU都正向，但未达到注册的实际增益及区间标准；后续文字实验共同配方保留R2。** Precision没有作为自动否决项。','',
        '| 种子 | 阶段 | 模型 | IoU | Dice | Precision | Recall | Brier |','|---|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        for side,name in (('baseline','R2'),('candidate','RS1')):
            d=row[side];lines.append(f"| {row['seed']} | {'发现' if row['seed']==1219 else '独立复验'} | {name} | "+' | '.join(f'{d[m]*100:.4f}%' for m in metrics[:-1])+f" | {d['brier']:.6f} |")
    lines+=['','| 种子 | ΔIoU pp | ΔDice pp | 最小面积组ΔIoU pp | FP总数变化 | FN总数变化 |','|---|---:|---:|---:|---:|---:|']
    for r in rows:
        c=r['pixel_counts_reconstructed_and_metrics_verified']['delta']
        lines.append(f"| {r['seed']} | {r['deltas']['iou']['mean']*100:+.4f} | {r['deltas']['dice']['mean']*100:+.4f} | {r['deltas']['iou']['small_mean']*100:+.4f} | {c['fp']:+d} | {c['fn']:+d} |")
    for scope,title in (('new_seeds','两个新种子2027/3407：主要复验'),('all_three_including_discovery','三个种子含1219：描述性汇总')):
        lines+=['','## '+title,'','均值±训练种子样本SD；IoU/Dice/Precision/Recall的分数均值与SD用百分数表示，配对差值用百分点，Brier用原始单位。','',
            '| 指标 | R2均值±SD | RS1均值±SD | 配对差值均值±SD |','|---|---:|---:|---:|']
        for m in metrics:
            scale=1 if m=='brier' else 100;d=stats[scope][m]
            lines.append('| '+m+' | '+' | '.join(f"{d[s]['mean']*scale:.6f} ± {d[s]['sample_sd']*scale:.6f}" for s in ('baseline','candidate','delta'))+' |')
    ci=[v*100 for v in summary['new_seed_grouped_ci']['ci95']]
    lines+=['','## 预注册判定','',
        f"新种子平均IoU差值 {summary['new_seed_mean_iou_delta']*100:+.4f} pp，描述性分组95%区间 [{ci[0]:+.4f}, {ci[1]:+.4f}] pp。",'',
        '| 条件 | 结果 |','|---|---|',
        f"| 两个新种子IoU均>0 | {summary['checks']['both_new_seeds_positive']} |",
        f"| 平均IoU提升≥0.3 pp | {summary['checks']['new_seed_mean_at_least_0_003']} |",
        f"| 分组描述性区间下界>0 | {summary['checks']['grouped_descriptive_ci_positive']} |",'',
        '1219是发现种子，不参与主要判定。三个种子的正向点估计支持进一步关注这个小幅配方收益，但不足以认领稳定Test收益，也不能据此放宽门槛。图像/患者组bootstrap不是跨训练种子显著性检验。', '',
        f"两个新种子的最小面积组平均IoU差值为 {stats['new_seeds']['iou']['small_mean_delta']*100:+.4f} pp；完整小面积指标见full_metric_summary.json。这是诊断，不追加为新的否决规则。像素总FP/FN与逐图macro是不同统计，不能互相替代。",'',
        '## 完成与后续','',
        '2027 Best68，末检距训练结束447.446秒；3407 Best74，末检距训练结束1169.774秒。每组检查2/2，均在半小时内完成，无持续SSH监控。两组分别在HF 758c6c54/、ec3d45cc/完成24文件的大小/Xet双端核验；各11份下载run文件亦一致。训练源、检查点、实际学习率、逐图指标与Train诊断状态恢复均已核验。','',
        '下一阶段以R2为共同配方，按TEXT_IMPLEMENTATION_PLAN.md先实现d3/up3后的T1视觉适配器、T2普通文字注意力，再分项验证短语绑定和图像证据控制。六份原R2源码已用Git对象核查并固定哈希；当前完成的是接口规格，未实现或训练新候选，不称第二创新点已经成立。']
    (HERE/'COMPLETE_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(verified=True,report='COMPLETE_REPORT.md',passed=summary['passed'],selected_common_recipe=summary['selected_common_recipe'],
        new_mean_iou_delta_pp=summary['new_seed_mean_iou_delta']*100,new_group_ci_pp=ci,
        new_small_mean_iou_delta_pp=stats['new_seeds']['iou']['small_mean_delta']*100)))

if __name__=='__main__':main()

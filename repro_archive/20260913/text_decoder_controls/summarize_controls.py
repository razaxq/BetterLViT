"""Reproduce the completed three-way control diagnosis from frozen Val exports.

No model execution, threshold search, training, or Test access occurs here.
Area strata describe errors after the experiment; they are not model inputs.
"""
import json
from pathlib import Path

import numpy as np

from analyze import METRICS, digest, validate_export
from remote_ops import DOCS, HERE, read, save


def main():
    paths = {
        'R2': DOCS / 'repro_archive/20260908/recipe_execution/r2_results/validation.json',
        'T1': HERE / 't1_results/validation.json',
        'T2': HERE / 't2_results/validation.json',
    }
    exports = {name: read(path) for name, path in paths.items()}
    records = {name: validate_export(data) for name, data in exports.items()}
    names = sorted(records['R2'])
    assert all(set(names) == set(rows) for rows in records.values())
    assert all(data['seed'] == 1219 for data in exports.values())
    for label in ('t1', 't2'):
        proof = read(HERE / (label + '_results/independent_verification.json'))
        assert proof['verified'] and not proof['threshold_reconciliation_needed']
        assert proof['inspection_number'] == 2 and proof['within_30_minutes']
    areas = np.array([records['R2'][name]['label_pixels'] for name in names])
    cutoffs = np.quantile(areas, [.25, .5, .75])
    strata = [areas <= cutoffs[0], (areas > cutoffs[0]) & (areas <= cutoffs[1]),
              (areas > cutoffs[1]) & (areas <= cutoffs[2]), areas > cutoffs[2]]
    assert np.all(np.stack(strata).sum(0) == 1)
    values = {name: {m: np.array([rows[n][m] for n in names]) for m in METRICS}
              for name, rows in records.items()}
    comparisons = {}
    for candidate, control in [('T1', 'R2'), ('T2', 'R2'), ('T2', 'T1')]:
        delta = {m: values[candidate][m] - values[control][m] for m in METRICS}
        by_area = []
        for index, mask in enumerate(strata):
            by_area.append(dict(quartile=index + 1, count=int(mask.sum()),
                area_min=int(areas[mask].min()), area_max=int(areas[mask].max()),
                metric_deltas={m: float(d[mask].mean()) for m, d in delta.items()},
                contribution_to_all_image_iou_delta=float(delta['iou'][mask].sum() / len(names))))
        assert abs(sum(x['contribution_to_all_image_iou_delta'] for x in by_area)
                   - delta['iou'].mean()) < 1e-12
        comparisons[candidate + '_minus_' + control] = dict(
            metric_deltas={m: float(d.mean()) for m, d in delta.items()},
            iou_improved=int((delta['iou'] > 1e-12).sum()),
            iou_worsened=int((delta['iou'] < -1e-12).sum()),
            iou_unchanged=int((abs(delta['iou']) <= 1e-12).sum()),
            area_strata=by_area,
            remaining_three_quartiles_iou_delta=float(delta['iou'][~strata[0]].mean()))
    pair = read(HERE / 't2_results/paired_comparison.json')
    pair_t1 = read(HERE / 't2_results/t2_vs_t1.json')
    direction_passed = (exports['T2']['macro_iou'] > exports['T1']['macro_iou']
                        and exports['T2']['macro_iou'] - exports['R2']['macro_iou'] >= .003)
    result = dict(verified=True, split='validation', test_split_accessed=False,
        discovery_seed_only=True, seed=1219, samples=len(names),
        source_artifacts={name: dict(path=path.relative_to(DOCS).as_posix(),
            sha256=digest(path), source_git_commit=exports[name]['checkpoint_git_commit'])
            for name, path in paths.items()},
        area_cutoffs=cutoffs.tolist(), comparisons=comparisons,
        t2_registered_extension_direction_passed=bool(direction_passed),
        retained_control='R2', new_training_started=False,
        interpretation='Post-hoc descriptive error stratification; no causal, Test, or seed-stability claim')
    save('completed_comparison.json', result)
    lines = ['# T0/T1/T2 完整对照与下一步诊断', '',
        '2026-09-13；三组均为原R2配方、80轮、seed1219、固定0.5阈值、1429张Val逐图macro指标。'
        'T0复用已完成R2；T1/T2已分别完成训练及Best导出。没有新增Test访问。', '',
        '| 组别 | 新增分支 | Best轮 | Val IoU | Val Dice | 相对R2 IoU |',
        '|---|---|---:|---:|---:|---:|']
    for name, content in [('R2', '无'), ('T1', '视觉注意力'), ('T2', '文字注意力')]:
        d = exports[name]
        lines.append(f"| {name} | {content} | {d['checkpoint_best_epoch']} | {100*d['macro_iou']:.4f}% | "
                     f"{100*d['macro_dice']:.4f}% | {100*(d['macro_iou']-exports['R2']['macro_iou']):+.4f} pp |")
    lines += ['', '**结论：保留R2。T2没有通过预先登记的推进方向，普通文字注意力未带来可确认的IoU收益。** '
        '这不证明所有文字机制都无效，也不支持把该模块认领为第二创新点。', '',
        f"T2−R2 IoU {100*pair['deltas']['iou']['mean']:+.4f} pp，分组描述性95%区间"
        f"[{100*pair['deltas']['iou']['ci95'][0]:+.4f}, {100*pair['deltas']['iou']['ci95'][1]:+.4f}] pp；"
        f"T2−T1 {100*pair_t1['deltas']['iou']['mean']:+.4f} pp，区间"
        f"[{100*pair_t1['deltas']['iou']['ci95'][0]:+.4f}, {100*pair_t1['deltas']['iou']['ci95'][1]:+.4f}] pp。"
        '两个IoU区间均跨零；这里是图像/患者分组抽样，不是独立训练种子检验。', '',
        '## 退步集中在哪里', '',
        '以下面积分组为完成实验后的描述性分析；按真值面积分组仅用于评估，不向模型提供真值或构造报告。', '',
        '| 面积分组 | 图像数 | 面积范围/像素 | T1−R2 IoU | T2−R2 IoU | T2−T1 IoU |',
        '|---|---:|---|---:|---:|---:|']
    for i, group in enumerate(comparisons['T2_minus_R2']['area_strata']):
        ds = [comparisons[k]['area_strata'][i]['metric_deltas']['iou'] * 100
              for k in ('T1_minus_R2', 'T2_minus_R2', 'T2_minus_T1')]
        lines.append(f"| Q{i+1} | {group['count']} | {group['area_min']}–{group['area_max']} | "
                     + ' | '.join(f'{d:+.4f} pp' for d in ds) + ' |')
    small = comparisons['T2_minus_R2']['area_strata'][0]
    rest = comparisons['T2_minus_R2']['remaining_three_quartiles_iou_delta']
    counts = pair['pixel_counts_reconstructed_and_metrics_verified']['delta']
    lines += ['', f"最小面积组贡献总体IoU变化{100*small['contribution_to_all_image_iou_delta']:+.4f} pp；"
        f"其余{len(names)-small['count']}张平均IoU变化{rest*100:+.4f} pp。"
        '不能将大病灶上的微小收益抵消小病灶伤害后描述为普遍改善。', '',
        f"T2相对R2，macro Precision {100*pair['deltas']['precision']['mean']:+.4f} pp，"
        f"Recall {100*pair['deltas']['recall']['mean']:+.4f} pp，Brier {pair['deltas']['brier']['mean']:+.8f}。"
        f"全体像素计数的TP增加{counts['tp']}、FP增加{counts['fp']}、FN减少{-counts['fn']}。"
        '这与过分割倾向一致；像素总计不能替代macro IoU，也不能单独确定文字内容是原因。', '',
        '## 分支有没有学到东西', '',
        'T2在Train第10/40/80轮各一个正常批次上的残差RMS/输入RMS约为0.166/0.197/0.188，'
        'Q投影梯度均非零。T1对应约0.076/0.093/0.095。记录支持这些批次上的分支已激活，'
        '不支持“仍然全零塌陷”的解释；它不能证明全数据定位正确，也不能证明T2残差较大是退步原因。'
        '两组有效key数不同，原始attention熵不能直接横比为定位质量。', '',
        '## 下一步顺序', '',
        '1. 先对冻结T2 Best做新增分支关闭、真实文字、同长度错配文字的Val诊断；原有文字通路始终使用正确报告。'
        '关闭分支后的结果属于同一T2权重的干预，不等于重新训练的R2。错配也不生成新的目标mask。',
        '2. 观察新增分支对小病灶FP和IoU的实际影响，再判断需要修正文字位置绑定还是限制文字影响。'
        '这些干预是事后机制诊断，不搜索阈值、不用错配下降直接认领创新、不把诊断最优设置当成正式结果。',
        '3. 再按已有计划分别验证T3短语绑定与T4图像证据控制，保持共同训练配方。'
        '当前三组已结束；本报告没有启动新的80轮训练。新实验须独立冻结源码、预检和存储审计。', '',
        '复现：`python repro_archive/20260913/text_decoder_controls/summarize_controls.py`。'
        '完整数值、输入SHA256及加权贡献恒等式核验见completed_comparison.json；分组bootstrap见各组paired_comparison.json。']
    (HERE / 'COMPLETE_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(verified=True, t2_registered_extension_direction_passed=bool(direction_passed),
                         comparisons=comparisons), ensure_ascii=False))


if __name__ == '__main__':
    main()

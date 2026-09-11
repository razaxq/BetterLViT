"""Render the verified, completed A run without accessing the server or dataset."""
from pathlib import Path
import json
import numpy as np

HERE = Path(__file__).resolve().parent
summary = json.loads((HERE/'results/summary.json').read_text())
rows = json.loads((HERE/'results/records.json').read_text())['records']
proof = json.loads((HERE/'results/independent_verification.json').read_text())
assert proof['verified']
state = json.loads((HERE/'state.json').read_text())
assert state['phase'] == 'complete' and state['inspections_completed'] == 1
lines = ['# 文字通路诊断 A：完成结果', '',
    '执行源 `da17b3fbef0ca4ac6343575cf2af18fab3c76efb`，标签 `diagnostic-r2-text-path-a-v3-20260911`。', '',
    '2026-09-11 悉尼22:31:38.326完成，首次预约检查在完成后33.862秒观察到结果；使用1/2次检查即关闭，不再检查本运行。32张Train CUDA预检及全部原文字身份核对通过。1429张Val的原文逐图指标与冻结R2完全相同，最大差0；模型状态/检查点不变，未访问Test。五份完整产物的字节/SHA256及全部计数、摘要已独立复算。', '',
    'R2原文 Val macro IoU **72.665359%**，Dice **82.403429%**。下列为错误或改写文字输入的诊断，不能称新方法的性能。', '',
    '| 文字条件 / 直接入口 | IoU (%) | 相对原文 (pp) | 95%图像配对区间 (pp) | token改变数 |',
    '|---|---:|---:|---|---:|']
for key, groups in sorted(summary['conditions'].items()):
    r = groups['all']; lo, hi = r['delta_iou_ci95']
    lines.append(f"| {key} | {100*r['mean']['iou']:.6f} | {100*r['delta_iou']:+.4f} | [{100*lo:+.4f}, {100*hi:+.4f}] | {r['tokens_changed']} |")
lines += ['', '## 控制措辞后的关系敏感性', '',
    '可交换关系的611张Val中包含315张单侧及296张非对称双侧。关系交换减canonical的IoU差：', '',
    '| 直接文字入口 | ΔIoU (pp) | 95%配对区间 (pp) | 平均逐图绝对软面积变化 (像素) |',
    '|---|---:|---|---:|']
selected = [r for r in rows if r['group'] in ('unilateral','asymmetric_bilateral')]
derived = {}
for scope, value in summary['binding_controlled_contrast'].items():
    r = value['swap_minus_semantic_canonical']['iou']; lo,hi = r['ci95']
    absolute = {metric: float(np.mean([abs(x['conditions']['relation_swap/'+scope][metric]-x['conditions']['canonical/'+scope][metric]) for x in selected])) for metric in ('soft_area','hard_area','fp','fn')}
    derived[scope] = absolute
    lines.append(f"| {scope} | {100*r['mean']:+.4f} | [{100*lo:+.4f}, {100*hi:+.4f}] | {absolute['soft_area']:.4f} |")
lines += ['', 'EPPA入口的关系交换影响明显超过主LViT入口。原文到同语义改写的影响较小，但两路同时canonical仍下降0.2537 pp，说明有一定措辞敏感性。原R2已经利用文字关系，不能把下一步简单包装为“让模型开始使用文字”。', '',
    '有符号平均软面积接近0不等于逐图保量：EPPA入口的swap−canonical平均为−0.5439像素，但逐图绝对变化均值为239.2583像素。FP/FN和面积不能只看有符号均值。', '',
    '主入口变化仍可能通过视觉/PLAM特征间接影响EPPA；这里隔离的是直接文字入口。通用文字属于信息减少且可能分布外的输入；诊断不能证明临床语义理解、因果定位或新结构IoU增益。图像bootstrap区间仅作描述，未校正多重比较。', '',
    '## 下一步', '',
    '进入已授权B：冻结R2，使用相同Train内部划分和特征，比较普通匹配/关系差分×有无软总量投影，并加入新增分支无文字及仅文字模板空间图控制。先注册协议并通过数值/梯度测试，再运行短程筛选。B的内部留出图像曾参与R2原始训练，不是独立泛化测试。官方Val/Test不用于B训练或筛选。A完成不自动触发80轮C，也不建立第二创新点。', '']
(HERE/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
(HERE/'area_absolute_analysis.json').write_text(json.dumps(derived,indent=2)+'\n',encoding='utf-8',newline='\n')

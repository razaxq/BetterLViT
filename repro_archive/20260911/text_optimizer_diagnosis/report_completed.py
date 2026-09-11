"""Render the registered Train-only optimizer control and its limits."""
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from analysis import write_json

HERE=Path(__file__).resolve().parent
s=json.loads((HERE/'results/summary.json').read_text())
t=json.loads((HERE/'results/train_diagnostics.json').read_text())
h=json.loads((HERE/'results/history.json').read_text())
state=json.loads((HERE/'state.json').read_text())
runtime=json.loads((HERE/'results/runtime.json').read_text())
assert json.loads((HERE/'results/independent_verification.json').read_text())['verified']
def stamp(ts):return datetime.fromtimestamp(ts,ZoneInfo('Australia/Sydney')).isoformat()
derived=dict(final_adamw_changes={},matched_loss_last128={})
for v in ('T1','T2','T3','T4','image','template'):
    rows=t[-1]['cases']['adamw/'+v]['records']
    derived['final_adamw_changes'][v]={metric:float(np.mean([r['output'][metric]-r['baseline'][metric] for r in rows])) for metric in
        ('iou','dice','precision','recall','brier','soft_area','hard_area','fp','fn')}
    derived['matched_loss_last128'][v]=float(np.mean([x['loss']['adamw/'+v]-x['loss']['adam_l2/'+v] for x in h[-128:]]))
write_json(HERE/'derived_analysis.json',derived)
lines=['# D：优化器塌缩对照完成结果','',
    '**结论：受控对照支持coupled L2导致T4/image弱分支塌缩；去掉衰减或改AdamW可以恢复可训练性，但恢复后没有带来新的IoU收益证据。停止把当前输出残差结构直接扩展到80轮。**','',
    f"18个案例各512步，{stamp(runtime['completed_unix'])}完成。首次检查距训练结束{state['training_completed_unix_inspection_delay_seconds']/60:.3f}分钟，距全部完成{state['completed_unix_inspection_delay_seconds']/60:.3f}分钟；检查1/2即关闭。5份原始结果SHA256、18×512步及32张fit逐图计数/摘要已独立核验。",'',
    '执行源`c7080ea82ecaca0e1f33df880d168e3eb7cbc6dd`，标签`diagnostic-text-optimizer-d1-20260911`。原Adam在0/256/512步的全部32张fit指标与B保存值精确复现，缓存/顺序/文字和原heads源码固定。未评估内部holdout、官方Val或Test，以下数值只描述参与拟合的固定32张Train诊断样本。基线IoU77.898719%。','',
    '## 512步后的诊断IoU变化（百分点）','',
    '| 新增头 | 原Adam+L2 | Adam无衰减 | AdamW |', '|---|---:|---:|---:|']
for v in ('T1','T2','T3','T4','image','template'):
    values=[100*s['cases'][method+'/'+v]['delta_iou'] for method in ('adam_l2','adam_zero_decay','adamw')]
    lines.append('| '+v+' | '+' | '.join(f'{x:+.4f}' for x in values)+' |')
lines += ['', 'T4无衰减/AdamW平均绝对残差恢复到0.13282，32/32张预测发生变化；原Adam仅2.6861e−15、0/32张变化。AdamW T4的K范数5.7981、末步任务梯度0.00649；原Adam为6.2186e−12、1.4264e−10。image同样由原Adam近零输出恢复为0.17843残差、32/32张变化。', '',
    '同一初始化、数据顺序、网络和loss下，仅改衰减机制便恢复学习，且原Adam准确复现旧结果，支持其为本配置塌缩的原因。该结论限于这些小头、seed和预算；不表示R2历史训练也受相同程度影响。AdamW在此很小衰减率下与无衰减结果接近，不能外推其对所有任务都更优。', '',
    '## 恢复学习后，修正仍偏向增加误检', '',
    '| AdamW新增头 | 平均FP变化（像素） | 平均FN变化（像素） | Precision变化（pp） | Recall变化（pp） |',
    '|---|---:|---:|---:|---:|']
for v,value in derived['final_adamw_changes'].items():
    lines.append(f"| {v} | {value['fp']:+.4f} | {value['fn']:+.4f} | {100*value['precision']:+.4f} | {100*value['recall']:+.4f} |")
lines += ['', 'T4增加20.16个FP，只减少4.75个FN；软总量变化约2e−7像素，但硬面积增加24.91像素，IoU下降0.2241 pp。T1/T2/T3分别下降0.5694/0.2691/0.0903 pp。模板头只有+0.0229 pp的极小fit变化，不能当作泛化收益或文字创新。', '',
    '相同最后128个训练批次中，AdamW T4的Dice/Focal loss比原Adam T4平均低0.0004893；固定32张fit的IoU却更低。这支持检查当前目标与二值IoU修正方向的相容性，但两项统计覆盖的样本不完全相同，不能据此证明所有样本存在loss-IoU冲突。B/D同时缺乏独立文字增益，继续增加分支复杂度没有实证理由。', '',
    '64步时AdamW T4曾在这32张fit上+0.0691 pp，到256/512步转负。这里不倒选64步作为成功checkpoint，也不根据这个事后观察调旧holdout。', '',
    '## 下一步', '',
    '归档本D的18头与完整负结果；原B失败门保持。下一步先做不更新模型的Train可行性审计，回答现有残差幅度/分辨率是否有足够纠错空间、目标梯度是否指向主要FP、以及真实文字提供的候选位置信息是否超过图像不确定性。详细可审查顺序见[NEXT_PLAN.md](NEXT_PLAN.md)。本报告不授权自动扩大到80/150轮或重新接触旧holdout/Test；后续方法需独立预注册并解决文字贡献的证据缺口。', '']
(HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
print('D completion report generated')

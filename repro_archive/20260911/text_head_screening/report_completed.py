"""Generate B's completed report from verified immutable artifacts."""
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from analysis import write_json

HERE=Path(__file__).resolve().parent
s=json.loads((HERE/'results/summary.json').read_text())
p=json.loads((HERE/'posthoc_collapse.json').read_text())
state=json.loads((HERE/'state.json').read_text())
runtime=json.loads((HERE/'results/runtime.json').read_text())
assert json.loads((HERE/'results/independent_verification.json').read_text())['verified']
def date(t):return datetime.fromtimestamp(t,ZoneInfo('Australia/Sydney')).isoformat()
lines=['# B六组文字残差头：完成报告','',
    '**结论：六组均无新增IoU增益；T3/T4均未通过预注册门，不启动80轮或Test。** T4与image有效输出塌缩为原R2，因此本轮不足以排除“关系差分+投影”在可训练配置下的可能性，也不能把T4高于退化T3当作创新增益。','',
    f"训练完成{date(runtime['training_completed_unix'])}，最终评估完成{date(runtime['completed_unix'])}。首次检查分别延迟{state['training_completed_unix_inspection_delay_seconds']/60:.3f}/{state['completed_unix_inspection_delay_seconds']/60:.3f}分钟，检查1/2即关闭。六组各1024步，九份原始结果逐文件SHA256和本地计数/摘要独立复算通过。",'',
    '执行源`49905dbbdc644454a37a4a49098db0d5df5fe75a`；标签`pilot-r2-text-residual-b-v1-20260911`。原R2所有5716张Train缓存概率与原生输出逐位相同，文字身份全量核验，基线状态/权重未变。下面是1131张Train内部留出，不是正式Val/Test；R2本身训练过这些图像，不能与A的1429张Val数值横比。','',
    '| 新增分支 | 内部IoU (%) | Dice (%) | ΔIoU (pp) | eligible ΔIoU (pp) | eligible group 95%区间 (pp) |',
    '|---|---:|---:|---:|---:|---|']
for name in ('baseline','T1','T2','T3','T4','image','template'):
    a=s['means'][name]
    if name=='baseline':delta=eligible=0.;ci=[0,0]
    else:
        c=s['contrasts'][name+'-baseline'];delta=c['all']['delta_iou'];eligible=c['eligible']['delta_iou'];ci=c['eligible']['group_ci95']
    lines.append(f"| {name} | {100*a['iou']:.6f} | {100*a['dice']:.6f} | {100*delta:+.4f} | {100*eligible:+.4f} | [{100*ci[0]:+.4f}, {100*ci[1]:+.4f}] |")
lines += ['', '## 误差方向及文字作用', '',
    '六组共同只在458张关系可交换样本启用新增修正；其他673张逐图保持R2。T1 eligible平均FP增加97.95像素、FN减少49.58像素，Precision下降1.5565 pp、IoU下降0.3783 pp。T3在软总量保持下仍平均增加38.96个FP、减少21.04个FN，硬面积增加60.00像素，IoU下降0.1601 pp；说明保软总量并不保证提升IoU或保持硬面积。', '',
    'T3最大软面积误差0.000176793像素，数值投影工作正常。T1/T3正确文字相对交换文字的eligible IoU分别−0.00533/−0.00245 pp，没有新增分支利用正确关系改善分割的证据。A表明整个R2使用文字，不能因此推断本新增头也利用了关系。', '',
    'T3内部Brier略改善但IoU/Dice/Precision退化。固定32张fit诊断的最终IoU相对R2：T1−0.6146、T2−0.1228、T3−0.1348、template+0.0951 pp。这是固定诊断子集，不是全fit泛化证据；至少说明问题不只表现为holdout过拟合。rare/common及小病灶完整分层见results/summary.json，不据小分层挑选正结果。', '',
    '## 输出塌缩与优化诊断', '',
    'T4 eligible平均绝对logit修正仅2.6033e−19，变化图像0/458；image修正为0，1024步中699步记录到总任务梯度范数0。T4最终梯度约9.88e−13。原数值预检证明算子可求导且在非零输出权重时激活，但没有验证弱分支在当前权重衰减下长期存活。', '',
    '后验诊断只读取小头权重及原首个16张fit批次，不更新参数、不重评holdout。CPU重建初始状态哈希与实际训练完全一致，检查点可加载且文件SHA/源/1024步匹配。', '',
    '| 参数 | T4初始范数 | T4最终范数 | image最终范数 |', '|---|---:|---:|---:|']
for name in p['variants']['T4']['initial']['parameters']:
    a=p['variants']['T4'];b=p['variants']['image']
    lines.append(f"| {name} | {a['initial']['parameters'][name]['parameter_norm']:.6g} | {a['final']['parameters'][name]['parameter_norm']:.6g} | {b['final']['parameters'][name]['parameter_norm']:.6g} |")
lines += ['', '零初始化的out使第一步Q/K/V/offset任务梯度为0，这是串联残差头的预期行为；但Adam的coupled L2仍直接对这些参数产生约3.1e−4至3.3e−4范数的更新梯度。后续弱分支参数几乎消失，而T1/T3的K也缩至原来的约1.5%/1.9%。这些事实支持“coupled L2压制弱分支”的解释，尚不是随机化优化器对照后的因果结论。预检中的非零有限梯度不能替代持续可训练性检查。', '',
    '## 接续动作', '',
    '先把本轮源/权重/完整负结果备份到HF短SHA目录并推送GitHub。下一步仅用原fit缓存做固定预算优化器诊断：原Adam+L2、Adam无衰减、AdamW解耦衰减。保留同一初始化、原序列前512步、原1024步余弦的前半段、原loss和全部六个头；比较参数存活、梯度、残差与固定32张fit诊断。它只检验塌缩解释，不重新访问1131张holdout、官方Val或Test，不更改本B的失败结论或阶段门。', '']
(HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
print('Completed B report generated')

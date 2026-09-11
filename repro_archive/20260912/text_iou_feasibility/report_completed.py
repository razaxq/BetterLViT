"""Write a factual E0/E1 report from independently verified stored results."""
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo
from analysis import write_json
HERE=Path(__file__).resolve().parent
R=HERE/'results';summary=json.loads((R/'summary_verified.json').read_text())
proof=json.loads((R/'independent_verification.json').read_text());assert proof['verified']
runtime=json.loads((R/'runtime.json').read_text());collection=json.loads((HERE/'collection.json').read_text())
finished=datetime.fromtimestamp(runtime['completed_unix'],ZoneInfo('Australia/Sydney')).isoformat()
delay=collection['observed_unix']-runtime['completed_unix']
all160=summary['cohorts']['all160'];additional=summary['cohorts']['additional128']
text=f'''# E0/E1：塌缩修复后，IoU 瓶颈在哪里

**结论：幅度上界没有排除改善空间，原 Dice/Focal 在基线输出处也没有把 FP/FN 的逐像素纠错方向推反。当前更值得处理的是粗空间修正与错误定位能力；现有 T4 在160张 fit 上仍为负，不能直接扩大到80轮。**

执行源 `{runtime['source_git_commit']}`，标签 `diagnostic-text-iou-e2-20260912`。{finished} 完成，计算约{runtime['completed_unix']-runtime['started_unix']:.2f}秒，预定收取时距完成{delay/60:.3f}分钟；一次远端完成收取，未保持SSH等待计算。全程0次参数更新，B缓存和D权重只读复用，未访问旧B holdout、官方Val或Test。6份原始JSON的SHA、1930张上界/160张逐图计数及转移、1920组梯度统计独立核验通过；原32张所有六头的全部指标精确复现D。28网格解析梯度与autograd校验通过。

这些样本都来自原R2训练集且属于D的fit部分。新增128只是在本诊断中预先按文件名哈希选出的额外样本，不是泛化验证集。以下IoU均是逐图macro、严格>0.5阈值。

## 1. E0：可跨阈值的错误仍不少

1930张eligible fit基线IoU为{summary['e0']['baseline_iou']*100:.6f}%。允许每像素都知道GT、独立选择最优残差，得到：

| GT oracle上界 | 可修复FP占全部FP | 可修复FN占全部FN | eligible平均IoU增量（pp） | 乘1930/4585的全fit折算（pp） |
|---|---:|---:|---:|---:|
'''
for key,label in (('unprojected','任意逐像素±0.5'),('projected_loose','投影的宽松±1上界')):
    s=summary['e0']['oracles'][key]
    text+=f"| {label} | {s['fp_correctable_fraction']*100:.2f}% | {s['fn_correctable_fraction']*100:.2f}% | {s['eligible_mean_gain']*100:.4f} | {s['all_fit_folded_gain']*100:.4f} |\n"
text+='''
上述上界使用GT且忽略空间平滑、视觉/文字表示的限制。投影上界还放松了联合保量可行性，仅用于排除“幅度太小必然无望”；不是可训练模型的成绩或Val/Test预期增益。无需据此立即把±0.5扩大。

## 2. E1：梯度方向先正确，空间约束会引入冲突

160张含107513个FP、46986个FN。下表是在同一基线概率处，实际Dice/Focal总目标的下降方向会抬高误检像素的前景概率、或压低漏检像素的前景概率的比例；不是分支参数之间的夹角统计，也不是训练结束后所有梯度的概括。

| 局部方向 | 全部FP有害方向比例 | 全部FN有害方向比例 | 近阈值FP有害方向比例 | 近阈值FN有害方向比例 |
|---|---:|---:|---:|---:|
'''
for key,label in (('unrestricted','独立逐像素'),('mass_projected','独立逐像素+保量切空间'),
    ('unrestricted_28grid','28网格双线性'),('mass_projected_28grid','28网格双线性+保量')):
    g=all160['gradients']['total/'+key]
    text+='| '+label+' | '+' | '.join(f"{g[k]['wrong_fraction']*100:.2f}%" for k in ('fp/all','fn/all','fp/near','fn/near'))+' |\n'
text+='''
近阈值定义|z|≤0.5。解析计算是可自由学习网格系数在初始identity处的方向，有投影时包含重新求lambda的有效logit变化；真实文字头参数共享/注意力还会附加限制。结果支持检验细尺度接口，但不能声称28网格解释了全部退化，或更高分辨率必然取得收益。多数错误的可行方向仍正确。

单点梯度方向正确也不保证有限步优化后的二值IoU提高。T4在相同10批/160图上，平均目标loss下降0.0002696，但macro IoU下降0.0678 pp；其中4批同时出现loss下降和IoU下降。这次是同样本比较，但仍是一个冻结头checkpoint的观察，不是换loss必然有效的证据。

## 3. E1：修正有用错误与破坏正确像素接近抵消

| 固定D AdamW头 | 原32 ΔIoU（pp） | 新增128 ΔIoU（pp） | 全160 ΔIoU（pp） |
|---|---:|---:|---:|
'''
for v in ('T1','T2','T3','T4','image','template'):
    text+='| '+v+' | '+' | '.join(f"{summary['cohorts'][c]['conditions'][v+'/correct']['delta']['iou']*100:+.4f}" for c in ('original32','additional128','all160'))+' |\n'
t=all160['conditions']['T4/correct'];tr=t['transitions']
text+=f'''
全160基线IoU {all160['baseline_iou']*100:.6f}%，T4为{t['mean']['iou']*100:.6f}%。每图平均纠正{tr['fixed_fp']:.2f}个FP和{tr['fixed_fn']:.2f}个FN，同时新增{tr['new_fp']:.2f}个FP和{tr['new_fn']:.2f}个FN。约{tr['fixed_fp']+tr['fixed_fn']:.2f}个原错被修好，却破坏约{tr['new_fp']+tr['new_fn']:.2f}个原正确像素。最后净FP+{t['delta']['fp']:.2f}、FN{t['delta']['fn']:.2f}、Precision{t['delta']['precision']*100:+.4f} pp。

当前小正数来自image/template控制的Train观察，不能称泛化增益。T4正确报告比关系交换报告IoU高{all160['correct_minus_swap_iou']['T4']*100:.4f} pp，说明新增分支有一定关系敏感性；但正确报告输出仍低于R2和这两个控制，**文字敏感不等于文字创造净收益**。错误报告只用于输入敏感性诊断，未更换GT，也不能当作另一个分割任务成绩。

## 4. E1：重点位置的错误富集弱于直接不确定性

每图固定前5%像素预算（2508点），按|p_head-p_base|选位置。T4所选位置中，原R2错误占{t['top5_error_precision']*100:.2f}%，直接按p(1-p)选点是{t['uncertainty_top5_error_precision']*100:.2f}%。image头为{all160['conditions']['image/correct']['top5_error_precision']*100:.2f}%。这是相同预算的原错误定位比较，不是纠正准确率，也不表明不确定性本身可以完成纠错。

因此不宜继续让当前T4差分直接决定在哪里修正。应先用共同的图像不确定性候选集验证细尺度纠错，再检验文字在候选处是否改善正确/错误判别。

## 5. 完成范围、失败记录和下一步

E0、同样本梯度、正确/关系交换文字、图像/模板定位对照已完成。**同义表达控制未完成**：固定160图的B缓存没有与canonical不同的source_text嵌入；没有新编码或把相同字符串重复推理冒充同义测试。新文字机制筛选必须补齐该控制。

首次源`5e98ac651a497fa1b3ee7de3949d3f3b0d0d8ce0`在精确复现断言处停止。只切换requires_grad标志的首16图只读对照显示，False触发约1e−11量级Brier差异、True与D全部指标完全相同；二值指标一致。v2保持D的标志且在no_grad内推理，无任何训练。原源/部分E0/失败日志及对照保留在attempt_v1；严格复现门没有放宽。

按[NEXT_PLAN.md](NEXT_PLAN.md)先验证细尺度接口，再验证同预算文字增量；暂不运行新80/150轮、不改Test阈值、不反复评旧holdout。第二创新点还未成立，本次提供的是更明确的可验证方向。源码及结果归档，不把优化器修复或普通局部精修写成新颖贡献。
'''
(HERE/'REPORT.md').write_text(text,encoding='utf-8',newline='\n')
write_json(HERE/'closure.json',dict(source_git_commit=runtime['source_git_commit'],phase='complete',train_updates=0,
    completed_sydney=finished,completion_collection_delay_seconds=delay,collection_within30min=0<=delay<=1800,
    remote_completion_collections=1,all_artifacts_verified=True,semantic_control_completed=False,
    new_training_started=False,official_validation_accessed=False,test_split_accessed=False))
print(json.dumps(dict(report=str(HERE/'REPORT.md'),finished=finished,delay_seconds=delay)))

"""Render the frozen decision and verified exports without altering any gate."""
from remote_ops import HERE,DOCS,read

d=read(HERE/'discovery_decision.json')
data=[('R2',read(DOCS/'repro_archive/20260908/recipe_execution/r2_results/validation.json'))]
data += [(label.upper(),read(HERE/(label+'_results/validation.json'))) for label in ('m1','m2')]
passed=d['discovery_gate_passed']
lines=['# M1/M2 发现阶段完整结果','',
    '结论：'+('通过单种子发现门，仍需额外种子和机制验证。' if passed else '未通过事前发现门，关闭本版本，不扩展多种子、不新增 Test 访问。'),
    '第二创新点尚未成立。以下均为同一 seed1219 的完整 Val 结果，不代表跨种子稳定收益。','',
    '## 完整指标','',
    '各组80轮、batch16；固定概率>0.5，1429张Val，逐图macro。R2复用历史导出，M1/M2的Best均为epoch67。',
    'EPPA V4-B、冻结CXR-BERT、无LoRA、Dice/Focal、boundary0、R2单次余弦及增强保持一致。','',
    '| 模型 | IoU | Dice | Precision | Recall | Brier |','|---|---:|---:|---:|---:|---:|']
for name,v in data:
    lines.append('| '+name+' | '+' | '.join(f"{v['macro_'+k]*100:.4f}%" for k in ('iou','dice','precision','recall'))+f" | {v['macro_brier']:.8f} |")
comparisons=[('M1−R2',read(HERE/'m1_results/paired_comparison.json')),('M2−R2',d['m2_vs_r2']),('M2−M1',d['m2_vs_m1'])]
lines += ['','## 配对差异','',
    '百分点差异；95%区间为10000次分组bootstrap的描述性区间，不是跨训练种子的显著性证明。','',
    '| 比较 | IoU差异 | IoU 95%区间 | Dice差异 | Precision差异 | Recall差异 |','|---|---:|---|---:|---:|---:|']
for name,c in comparisons:
    v=c['deltas'];ci=v['iou']['ci95']
    lines.append(f"| {name} | {v['iou']['mean']*100:+.4f} | [{ci[0]*100:+.4f}, {ci[1]*100:+.4f}] | "+' | '.join(f"{v[k]['mean']*100:+.4f}" for k in ('dice','precision','recall'))+' |')
lines += ['','## 小病灶与像素计数','',
    '最小mask四分位按GT面积固定：358张，面积<=2075像素。像素计数用于解释预测变化，不替代macro指标。','',
    '| 比较 | 小病灶IoU差异(pp) | Dice差异(pp) | Precision差异(pp) | Recall差异(pp) | 总体FP变化 | 总体FN变化 |',
    '|---|---:|---:|---:|---:|---:|---:|']
for name,c in comparisons:
    counts=c['pixel_counts_reconstructed_and_metrics_verified']['delta']
    lines.append('| '+name+' | '+' | '.join(f"{c['deltas'][k]['small_mean']*100:+.4f}" for k in ('iou','dice','precision','recall'))+f" | {counts['fp']:+d} | {counts['fn']:+d} |")
lines += ['','## 事前判据逐项结果','', '| 判据 | 通过 |','|---|---|']
labels={
    'm2_vs_r2_iou_at_least_point003':'M2−R2 IoU至少+0.30pp',
    'm2_vs_r2_iou_grouped_ci_lower_positive':'M2−R2 IoU区间下限>0',
    'm2_vs_m1_iou_positive':'M2−M1 IoU>0',
    'm2_vs_m1_iou_grouped_ci_lower_positive':'M2−M1 IoU区间下限>0',
    'm2_vs_r2_dice_nonregression':'M2对R2 Dice不退化',
    'm2_vs_m1_dice_nonregression':'M2对M1 Dice不退化',
    'm2_vs_r2_small_iou_nonregression':'M2对R2小病灶IoU不退化',
    'm2_vs_m1_small_iou_nonregression':'M2对M1小病灶IoU不退化'}
for k,ok in d['gates'].items():lines.append(f"| {labels[k]} | {'是' if ok else '否'} |")
lines += ['','## 机制解释与边界','',
    'M1的总体IoU点估计略高于R2，但区间跨0且小病灶IoU未提升。M2没有胜过M1，不能把这次视觉聚合的结果归因于新增文字锚点。',
    'M2的Recall更高，但Precision降低、FP增加，小病灶指标变差；把新增值向量限制为图像，并未在本配置中避免预测范围扩大的表现。这不证明所有视觉值聚合方法无效，也不单独确定退化的因果机制。',
    '第10、40、80轮采样Train批次的新分支有非零残差与Q/K/V梯度，说明其参与训练；这些稀疏观察不能证明正确定位或有益作用。',
    '原型聚合和语言条件本身不能作为创新声明，相关先例和差异化要求见预注册PROTOCOL.md。当前不以调阈值、延长训练、事后改门槛或更名挽救本版本。','',
    '## 来源与执行核验','']
sources=read(HERE/'sources.json')
for label in ('m1','m2'):
    p=read(HERE/(label+'_results/independent_verification.json'));h=read(HERE/(label+'_results/hf_upload_verified.json'))
    assert p['verified'] and not p['threshold_reconciliation_needed'] and h['verified']
    lines += [f"- {label.upper()}源 `{sources[label]['source_git_commit']}`，tag `{sources[label]['experiment_tag']}`；检查{p['inspection_number']}/2，末检距训练结束{p['seconds_after_training_end']:.3f}秒。",
              f"- HF `{h['bucket']}/{h['bucket_prefix']}/`：{h['verified_files']}文件、{h['logical_bytes']}字节，双端size/Xet及本地下载证据核验通过；Best/Last保留。"]
lines += ['','两组均完成80轮，实际LR、Best选择、源码、整数计数macro和四次Train观察核验通过，训练与Val退出码均0。未新增Test访问。',
    '冻结的门槛计算结果见discovery_decision.json；各组原始导出、日志、配对比较和备份回执保留。']
(HERE/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
print('RESULTS_ZH.md written; registered gates unchanged')

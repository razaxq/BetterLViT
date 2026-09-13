"""Summarize implementation, verified preflight and dispatch without claiming results."""
from remote_ops import HERE,DOCS,read
sources=read(HERE/'sources.json');preflight=read(HERE/'preflight_verified.json')
lines=['# 中间解码层文字对照：执行状态','',
    '共同训练配方保持R2，IoU优先。普通文字注意力是后续文字创新的必要对照，本身不作为第二创新点。','',
    '| 组别 | 内容 | 预算/来源 | 当前状态 |','|---|---|---|---|',
    '| T0 | 原R2，无新增分支 | 已完成80轮，seed1219，9eca26de5b301099805530edbf5a1a8718bea662 | 复用匹配对照；Val IoU72.6654%、Dice82.4034% |']
for label,content in [('t1','新增视觉context attention'),('t2','新增文字context attention')]:
    state_path=HERE/(label+'_state.json')
    status='已实现、冻结、预检通过；尚未启动'
    if state_path.exists():
        s=read(state_path)
        status=f"已提交；最近确认阶段{s.get('last_phase','dispatch')}；已检查{s['inspections_completed']}/2"
    result_path=HERE/(label+'_results/independent_verification.json')
    if result_path.exists():
        v=read(result_path)
        if v['verified']:
            status=f"80轮及Val已完成，Best{v['best_epoch']}；已检查{v['inspection_number']}/2"
    lines.append(f"| {label.upper()} | {content}，16896参数 | 80轮，seed1219，`{sources[label]['source_git_commit']}` | {status} |")
lines+=['','两组均在up3输出128×56×56后、up2前加入同容量分支；4头、宽32、32个context token。T1不读取报告长度；T2屏蔽padding，空报告回到零残差。原EPPA、原文字路径、legacy增强、Dice/Focal、无LoRA、80轮单次余弦均保持。','',
    '已验证：CPU masking/空报告/梯度检查；真实Train batch16五步；原R2与关闭分支后的完整五步grouped Adam输出/loss精确一致；两候选原模型权重、输入、CPU/CUDA RNG、第一次预测及适配器初始化分别精确匹配。观察开关开/关五步loss、输出、梯度完全相同。候选参数均16896、实测接口B×128×56×56和B×32×128、峰值分配约15.44GiB。短步loss不作为IoU增益证据。','',
    '执行顺序：T1完整80轮与自动Best验证集导出，核验、备份、发布后接T2完整80轮，不按T1早期排名取消T2。比较T1−T0、T2−T0、T2−T1，逐图macro IoU优先，报告Dice、小病灶与分组bootstrap。预检未访问Val/Test；正式运行中自动使用Val选择Best并导出逐图指标。本轮筛选不新增Test访问，最终性能主张仍需冻结候选后的Test结果。']
if (HERE/'t1_launch.json').exists():
    s=read(HERE/'t1_launch.json')
    lines+=['',f"T1提交历史：{s['started_sydney']}；提交时初估结束：{s['initial_predicted_training_end_sydney']}；首次预测检查预约：{s['planned_first_check_sydney']}。此段保留提交回执和最初预测，后续实测快照以下述检查记录为准。"]
for label in ('t1','t2'):
    path=HERE/(label+'_first_verified.json')
    if path.exists():
        v=read(path);f=v['forecast'];progress=v['latest_logged_train_progress']
        detail=f"第{progress[0]}轮{progress[1]}/{progress[2]} batch" if progress else '日志无可解析的新轮进度'
        lines+=['',f"{label.upper()}首检历史：{v['checked_sydney']}确认完成{v['completed_epochs']}/80轮，{detail}，GPU {v['gpu']}；来源、manifest和跟踪文件一致，日志尾部未见致命错误。当时使用1/2检查，最近完整post-warmup轮平均{f['mean_epoch_seconds']:.6f}秒，预测训练结束{f['predicted_training_end_sydney']}；唯一末检预约{f['final_check_sydney']}。这是训练健康和时间预测记录，后续终态以下述完成核验为准。"]
for label in ('t1','t2'):
    path=HERE/(label+'_results/independent_verification.json')
    if path.exists():
        proof=read(path);assert proof['verified']
        result=read(HERE/(label+'_results/validation.json'))
        pair=read(HERE/(label+'_results/paired_comparison.json'))
        snapshot=read(HERE/(label+'_final_snapshot.json'))
        ci=pair['deltas']['iou']['ci95']
        lines+=['',f"{label.upper()}完成核验：80轮、Best{result['checkpoint_best_epoch']}，1429张Val的macro IoU **{result['macro_iou']*100:.4f}%**、Dice **{result['macro_dice']*100:.4f}%**；相对R2 IoU {pair['deltas']['iou']['mean']*100:+.4f} pp，Dice {pair['deltas']['dice']['mean']*100:+.4f} pp，IoU分组描述性95%区间[{ci[0]*100:+.4f},{ci[1]*100:+.4f}] pp。最小面积{pair['small_count']}张IoU变化{pair['deltas']['iou']['small_mean']*100:+.4f} pp。末检{snapshot['checked_sydney']}，距训练结束{proof['seconds_after_training_end']:.3f}秒，检查{proof['inspection_number']}/2，半小时内={proof['within_30_minutes']}。全部轮学习率、Best来源、逐图整数计数与macro均核验；单种子Val不能证明稳定Test增益。详见{label}_results/REPORT.md和paired_comparison.json。"]
        backup=HERE/(label+'_results/hf_upload_verified.json')
        if backup.exists():
            b=read(backup)
            lines+=['',f"{label.upper()} HF备份：{b['bucket']}/{b['bucket_prefix']}/，{b['verified_files']}文件、{b['logical_bytes']}字节双端size/Xet核验，原Best/Last保留。备份核验时训练盘可用{b['scratch_free_bytes']}字节，shared实际{b['fs_bytes']}字节。"]
if (HERE/'t2_launch.json').exists():
    s=read(HERE/'t2_launch.json')
    lines+=['',f"T2提交历史：{s['started_sydney']}提交80轮训练，来源保持冻结的488ef093de80df71ee77741a8c7ee7b938c7d6b5；提交时初估结束{s['initial_predicted_training_end_sydney']}，首次检查预约{s['planned_first_check_sydney']}。此段保留提交回执与最初预测，后续实测以上述检查快照为准。"]
if (HERE/'storage_cleanup.json').exists():
    c=read(HERE/'storage_cleanup.json')
    proofs=[read(p) for p in sorted((HERE/'historical_backup').glob('*_verified.json'))]
    total=sum(f['bytes'] for p in proofs for f in p['files'])
    lines+=['',f"启动前存储清理历史：补齐C0/P1/P2/P3四个历史pilot的source、log、Best/Last和TB，共{sum(p['verified_files'] for p in proofs)}文件、{total}字节，按源短SHA双端size/Xet核验。这是已完成历史训练的检查点归档，没有新增或重新认证其评估。连同已备份P11，删除5个无活动引用的旧Last副本，释放{c['logical_bytes']}字节；当时训练盘可用{c['before']['scratch_free_bytes']}→{c['after']['scratch_free_bytes']}字节；shared实际{c['after']['shared_bytes']}字节，小于20GB。所有Best、近期RS1 Best/Last、F/B缓存、数据与环境保留。"]
lines+=['','源分支和独立实验tag已在GitHub核对；代码与执行证据一并保存。详见PROTOCOL.md、sources.json、preflight_verified.json、storage_cleanup.json及原生定时核验回执。']
completed=HERE/'completed_comparison.json'
if completed.exists():
    c=read(completed);assert c['verified']
    delta=c['comparisons']['T2_minus_R2']['metric_deltas']['iou']*100
    delta_t1=c['comparisons']['T2_minus_T1']['metric_deltas']['iou']*100
    lines+=['',f'本批T0/T1/T2对照已结束：T2−R2 IoU {delta:+.4f} pp，T2−T1 {delta_t1:+.4f} pp，未通过预登记的文字推进方向。继续保留R2；先做冻结分支干预诊断，再决定T3/T4实现。完整三组表、面积分组和证据边界见COMPLETE_REPORT.md；当前没有新训练在运行。']
closed=HERE/'batch_closed_verified.json'
if closed.exists():
    c=read(closed);assert c['verified'] and c['automation_config_absent']
    lines+=['','本批原生heartbeat `betterlvit` 已删除，工具回执与本地配置消失已核验，不再重复检查已完成训练。上述首次/末次预约均为历史执行记录。']
(HERE/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
tracker=DOCS/'docs/EXPERIMENT_TRACKER.md';text=tracker.read_text(encoding='utf-8')
heading='## 中间解码层文字对照 T1/T2（2026-09-13）'
section=heading+'\n\n'+'\n'.join(lines[2:])+'\n\n[完整执行档案](../repro_archive/20260913/text_decoder_controls/README.md)。\n\n'
if heading in text:
    start=text.index(heading);end=text.index('\n## ',start+len(heading));text=text[:start]+section+text[end+1:]
else:
    text=text.replace('## IoU优先：RS1独立种子复验（2026-09-12）',section+'## IoU优先：RS1独立种子复验（2026-09-12）',1)
text=text.replace('本批两组均80轮、各检查2/2、结束后半小时内末检，完整结果和Best/Last已分别在HF短SHA目录双端核验；无新训练提交。下一阶段已完成',
    'RS1本批两组均80轮、各检查2/2、结束后半小时内末检，完整结果和Best/Last已分别在HF短SHA目录双端核验。以下为T1/T2实现前的规格历史记录；当前状态以上方新阶段为准。当时下一阶段已完成')
text=text.replace('当前只有实现规格及源码审计，**尚未实现或训练新候选**','当时只有实现规格及源码审计，**尚未实现或训练新候选**')
text=text.replace('原生heartbeat `betterlvit` 已暂停并核验实际状态为PAUSED，不再重复旧实验检查；','RS1结束时原生heartbeat `betterlvit` 已暂停并核验为PAUSED；新T1/T2定时以后续新回执为准，不再重复旧实验检查；')
tracker.write_text(text,encoding='utf-8',newline='\n')

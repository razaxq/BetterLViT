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
    lines.append(f"| {label.upper()} | {content}，16896参数 | 80轮，seed1219，`{sources[label]['source_git_commit']}` | {status} |")
lines+=['','两组均在up3输出128×56×56后、up2前加入同容量分支；4头、宽32、32个context token。T1不读取报告长度；T2屏蔽padding，空报告回到零残差。原EPPA、原文字路径、legacy增强、Dice/Focal、无LoRA、80轮单次余弦均保持。','',
    '已验证：CPU masking/空报告/梯度检查；真实Train batch16五步；原R2与关闭分支后的完整五步grouped Adam输出/loss精确一致；两候选原模型权重、输入、CPU/CUDA RNG、第一次预测及适配器初始化分别精确匹配。观察开关开/关五步loss、输出、梯度完全相同。候选参数均16896、实测接口B×128×56×56和B×32×128、峰值分配约15.44GiB。短步loss不作为IoU增益证据。','',
    '执行顺序：T1完整80轮与自动Best验证集导出，核验、备份、发布后接T2完整80轮，不按T1早期排名取消T2。先比较T1−T0、T2−T0、T2−T1，逐图macro IoU优先，报告Dice、小病灶与分组bootstrap。当前预检未访问Val/Test，尚无新模型的完整IoU结果；本轮筛选不新增Test访问，最终性能主张仍需冻结候选后的Test结果。']
if (HERE/'t1_launch.json').exists():
    s=read(HERE/'t1_launch.json')
    lines+=['',f"T1提交历史：{s['started_sydney']}；提交时初估结束：{s['initial_predicted_training_end_sydney']}；首次预测检查预约：{s['planned_first_check_sydney']}。此段保留提交回执和最初预测，后续实测快照以下述检查记录为准。"]
for label in ('t1','t2'):
    path=HERE/(label+'_first_verified.json')
    if path.exists():
        v=read(path);f=v['forecast'];progress=v['latest_logged_train_progress']
        detail=f"第{progress[0]}轮{progress[1]}/{progress[2]} batch" if progress else '日志无可解析的新轮进度'
        lines+=['',f"{label.upper()}首检：{v['checked_sydney']}确认完成{v['completed_epochs']}/80轮，{detail}，GPU {v['gpu']}；来源、manifest和跟踪文件一致，日志尾部未见致命错误。已使用1/2检查，最近完整post-warmup轮平均{f['mean_epoch_seconds']:.6f}秒，预测训练结束{f['predicted_training_end_sydney']}；唯一末检预约{f['final_check_sydney']}。这是训练健康和时间预测记录，不是完成结果或IoU增益证据；实际结束到检查间隔需末检核验。"]
if (HERE/'storage_cleanup.json').exists():
    c=read(HERE/'storage_cleanup.json')
    proofs=[read(p) for p in sorted((HERE/'historical_backup').glob('*_verified.json'))]
    total=sum(f['bytes'] for p in proofs for f in p['files'])
    lines+=['',f"存储：补齐C0/P1/P2/P3四个历史pilot的source、log、Best/Last和TB，共{sum(p['verified_files'] for p in proofs)}文件、{total}字节，按源短SHA双端size/Xet核验。这是已完成历史训练的检查点归档，没有新增或重新认证其评估。连同已备份P11，删除5个无活动引用的旧Last副本，释放{c['logical_bytes']}字节；训练盘可用{c['before']['scratch_free_bytes']}→{c['after']['scratch_free_bytes']}字节；shared实际{c['after']['shared_bytes']}字节，小于20GB。所有Best、近期RS1 Best/Last、F/B缓存、数据与环境保留。"]
lines+=['','源分支和独立实验tag已在GitHub核对；代码与执行证据一并保存。详见PROTOCOL.md、sources.json、preflight_verified.json、storage_cleanup.json及原生定时核验回执。']
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

> 最终状态（2026-09-18 21:48 UTC）：12/12模型及全部Val/Test完成并核验，巡检已暂停并回读确认，无后续检查；完整结果见[最终报告](FINAL_REPORT.md)。以下为历史计划及巡检记录。

> 最新状态（2026-09-18 18:36 UTC）：11/12完成；仅FSDR seed3407 Test IoU75.4578%，三个种子仅FSDR平均75.4072±0.2614%。最后cr2s3407完成6/80轮，下次预计北京时间9月19日05:46，实际保存安排已回读核验。详见[本次报告](MONITOR_REPORT_20260918T1836.md)。以下保留历史状态。

> 最新状态（2026-09-18 13:46 UTC）：seed2027四组完成，PLAM＋完整RACE Test IoU75.3045%、较PLAM-0.0144个百分点；cr1s3407已自动接续，完成9/80轮。下一次约北京时间9月19日02:31，安排已回读核验。详见[本次报告](MONITOR_REPORT_20260918T1346.md)。以下保留历史状态。

> 最新状态（2026-09-18 10:06 UTC）：仅FSDR seed2027完成，Test IoU75.1242%，比PLAM低0.1948个百分点；完整RACE在FSDR上提升0.2439个百分点。cr2s2027已接续，完成6/80轮。下一次约北京时间9月18日21:20，已回读核验。详见[本次报告](MONITOR_REPORT_20260918T1006.md)。以下保留历史状态。

> 最新状态（2026-09-18 06:22 UTC）：seed1219四组完成，PLAM＋完整RACE Test IoU75.3830%，较PLAM+0.2921个百分点；cr1s2027已自动接续，完成25/80轮。下次预计北京时间9月18日18:04，安排已回读核验。详见[本次报告](MONITOR_REPORT_20260918T0622.md)。以下保留历史状态。

> 最新状态（2026-09-18 01:46 UTC）：仅FSDR seed1219完成，Test IoU75.6396%；完整RACE在FSDR上的同seed IoU差值-0.0040个百分点。cr2s1219已接续，完成9/80轮。下次预计北京时间9月18日12:52，保存安排已核验。详见[本次报告](MONITOR_REPORT_20260918T0146.md)。以下保留历史记录。

> 最新状态（2026-09-17 20:57 UTC）：前置六模型全部完成，三种子完整组合平均Test IoU增量+0.4987个百分点；cr1s1219已自动接续并完成11/80轮。下一次约北京时间9月18日09:44，保存安排已回读核验。详见[本次报告](MONITOR_REPORT_20260917T2057.md)。以下部署说明保留历史状态。

# FSDR × 完整RACE：余弦热重启三种子整体消融扩展

授权：用户2026-09-18要求“当前训练完成后继续拓展RACE的消融实验，和之前一样”。沿用先前四组整体消融；RACE路由与绑定修复辅助监督作为一个完整模块，不安排内部拆分。

| 配置 | 解码模块 | 完整RACE | 种子 | 本轮动作 |
|---|---|---|---|---|
| CR0 | 原PLAM，无FSDR | 关闭 | 1219/2027/3407 | 复用三组已完成结果 |
| CR1 | FSDR | 关闭 | 1219/2027/3407 | 新增三次训练 |
| CR2 | 原PLAM，无FSDR | 开启 | 1219/2027/3407 | 新增三次训练 |
| CR3 | FSDR | 开启 | 1219/2027/3407 | 复用两组结果，并等待当前3407完成 |

新增六次串行顺序：cr1s1219→cr2s1219→cr1s2027→cr2s2027→cr1s3407→cr2s3407。最终12模型。主模块修改的理由：补齐同配方、同种子的两个缺失对照，分别估计RACE在PLAM/FSDR上的增量以及两个模块的交互；模型实现没有新增修改。

共同设置完全沿用seed1219：QaTa-COV19-v2，Train5716/Val1429/Test2113；224×224，batch16，80轮；冻结CXR-BERT，32token，无LoRA；Dice/Focal各0.5、gamma2；Adam、weight decay1e-4及原参数分组；原数据增强；无边界损失、TCSR、RACE-PE或额外视觉预训练编码器。余弦热重启初始3e-4、eta_min1e-4、T_0=10、T_mult=1，每轮验证后step。没有恢复旧checkpoint，也不改变损失或选模方式。

完整RACE包含路由与绑定修复辅助监督，辅助权重0.05；报告/视觉/一致性权重0.4/0.4/0.2；报告项0.75六区域BCE＋0.25计数CE。PLAM保持既定公共参数随机初始化对齐，不加载已训练FSDR权重。

本次CR1显式关闭无效RACE字段（route/binding=false、aux=0），模型本身无RACE分支；与历史仅FSDR参考的初始化、前向输出和损失逐位一致性由真实Train-only前后向预检确认。CR2沿用历史PLAM＋完整RACE，只换热重启；对应历史同种子J2预检指纹严格对齐。旧profiles保持原样，评估循环与模型文件无改动。

已在服务器部署六个独立工作树、完整40位源码SHA及实验标签。六配置CPU实际80轮学习率、非法配置拒绝、导出CLI、评估循环一致性检查通过，队列等待/失败/来源校验测试通过。GPU真实Train前后向尚待当前实验结束后逐个执行；通过后才训练，不在当前GPU训练期间执行。

服务器接续进程PID588632已启动，状态waiting_dependency；先读取前置cos_restart_three_seeds_20260917队列调度状态，只有四个新run训练、Val和Test全部成功、来源一致且GPU空闲才释放新任务。每组预检→80轮训练→Val macro IoU选Best→一次同Best Test→下一组，失败即停；不自动重启失败训练，不按Test分数调整配方。队列独立运行，不依赖定期检查唤醒。等待期间每60秒仅读调度状态，不检查训练日志。

存储：新checkpoint及训练会话输出放在/root/autodl-fs/cos_restart_overall_20260918/<run>/Covid19，实际解析为/autodl-fs/data/cos_restart_overall_20260918/<run>/Covid19；独立仓库Covid19目录通过软链指向这里。部署时该盘可用196078129152字节，训练盘可用12631089152字节。没有清理或删除任何权重，保留全部新旧Best/Last用于预测图。挂载盘存储不等于已核验HF备份，不宣称新权重已备份HF。

监控：继续同一fsdr-race，前置cr3s3407剩余一次末检，新增六模型各最多两次主动检查，当前均0/2。首次按最多8个完整epoch中位耗时估计结束，末检加15分钟评估缓冲；检查前持久化计数，失败计数，不重置、不追加第三次。前置原目标2026-09-17T20:21Z（北京时间9月18日04:21）保持。下一次收齐旧六模型并运行旧目录finalize_summary.py，但不能暂停任务；接着首次检查新队列实际active。以后逐组末检＋后组首检，更新同一自动任务后view及回读实际保存配置。

新目录工具：read_queue.py只读调度；inspect_status.py是唯一主动进度检查入口；sync_completed.py收取完成的runtime/Val/Test/preflight和哈希；finalize_summary.py在全部完成后汇总12模型。禁止重复运行prepare_extension.py、deploy.py、launch_queue.py。

最终报告四组Test逐图macro IoU/Dice均值±样本标准差、全部种子结果，以及逐种子CR1−CR0、CR2−CR0、CR3−CR1、CR3−CR2、CR3−CR0；交互项CR3−CR1−CR2＋CR0。另列各组同种子单周期余弦对照，分清调度差异和模块贡献。不挑种子、不宣称统计显著性。12模型结果与保存状态完成后才暂停heartbeat，并view及回读确认。

数据划分仍沿用既有Train/Val间434个可恢复患者ID重叠及匿名样本未完全核验的限制，不宣称完全患者独立划分或官方LViT严格复现。

## 冻结来源

| run | 源码SHA | tag |
|---|---|---|
| cr1s1219 | 736fc717daf3a4ba1fa9645516efa80bba22909f | experiment-cos-restart-cr1-80e-seed1219-20260918 |
| cr2s1219 | ceaf626c775d309ca6af8ecd2c80a0dc75bd0f33 | experiment-cos-restart-cr2-80e-seed1219-20260918 |
| cr1s2027 | 7babb7a59e77d35c9e8221da5c552841dd584ea4 | experiment-cos-restart-cr1-80e-seed2027-20260918 |
| cr2s2027 | a2f439337bc5249de2e98d9e5f471af5e551795e | experiment-cos-restart-cr2-80e-seed2027-20260918 |
| cr1s3407 | 512fe59bfe3dd23e8441da8b43ead972e647f3f7 | experiment-cos-restart-cr1-80e-seed3407-20260918 |
| cr2s3407 | 3edb21f6c81e9e3248d8a3429e937c77850cce0c | experiment-cos-restart-cr2-80e-seed3407-20260918 |

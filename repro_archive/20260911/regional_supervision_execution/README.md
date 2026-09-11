# RS1 → RS2 → RS3执行入口

本轮用户于2026-09-11明确要求“启动训练”。固定执行三个首轮候选，各80轮、seed1219、batch16、R2单次余弦、原Dice/Focal。RS1整图软IoU，RS2最终输出的56窗口/28步长局部软IoU，RS3仅改变为阳性/背景分组。按[阶段计划](../../../docs/REGIONAL_SUPERVISION_EXPERIMENT_PLAN_20260911.md)推进，不按前一组数值修改后一组。

## 当前状态：三组全部完成、独立核验并备份

RS3于悉尼2026-09-11 **21:48:32.654**完成80轮、**21:50:19.998**完成1429张Val，Best67。**22:01:44.563**唯一末检距训练结束791.909秒（13.20分钟），检查2/2；训练/Val退出码0、来源及跟踪文件干净，实际80轮学习率、逐图指标与四次Train诊断状态恢复均通过独立核验。没有Test访问，训练历史与逐图导出IoU差−3.94e−9，在原界限内，不需要追加指标诊断。

RS3 Val IoU **72.8557%**、Dice **82.5468%**，相对R2 IoU **+0.1903 pp**、95%区间[−0.0668,+0.4506]，未通过IoU最低增量、区间及Precision三项门。RS3−RS2仅**+0.0162 pp**、区间[−0.2626,+0.3069]，没有证明分组有独立收益。详见[RS3报告](rs3_results/REPORT.md)及[三组汇总](FIRST_ROUND_REPORT.md)。RS1的单种子Val IoU正向证据仍保留，不能将综合门失败写成没有任何IoU增加。

RS3已追加备份到HF Bucket `razaxq/BetterLViT/f7984233/`：**21文件、1,694,438,969字节**，远端及独立本地清单路径/大小/Xet一致，11项下载run产物也匹配。Best/Last保留，共享fs **18,559,782,256字节**，训练盘余 **2,567,913,472字节**。未清理或删除模型。完成三组首轮后按用户新授权接续[文字诊断A](../text_grounding_execution/README.md)，同一heartbeat继续使用，不启动RS4或延长150轮。

## RS3历史启动与首检记录

RS3于悉尼**2026-09-11 17:13:13.659**完成首次短连接检查：完整4/80轮，第5轮至少340/357训练批次，GPU100%、显存17402MiB，来源SHA/manifest一致且跟踪文件干净，日志尾部无异常。尚无完整RS3结果，未访问Test，检查预算1/2。第2—4轮平均217.763331秒，加120秒后续Train诊断余量，预测**今天21:47:50悉尼时间**训练结束。

同一heartbeat `betterlvit` 已改约**今天22:00悉尼时间唯一末检**，保存的rrule、目标thread、提示及预算均核验。之后不再中途连接；末检时核对真实结束至检查是否≤30分钟，不能把预测当成已完成。见`rs3_first_snapshot.json`、`rs3_final_automation_verified.json`。

原始RS3提交为16:55:06.973，PID261154，来源`f79842331e4e5e41526ed66da31ec174ea4a61b2`，tag `experiment-rs3-regional-80e-seed1219-20260911`。80轮、seed1219、λ=0.128312，局部窗口阳性/空背景分组归一化；启动前训练盘可用4,284,715,008字节，fs18,559,782,256字节。原17:11首检预约保留于`rs3_automation_verified.json`；实际检查17:13:13，后续按新末检预约执行。

RS2于悉尼**2026-09-11 16:36:27.368**完成80轮，**16:38:13.553**完成1429张Val导出；Best75，训练/Val退出码均0。唯一末检**16:46:13.188**距训练结束585.820秒（9.76分钟），符合≤30分钟，检查预算2/2已用完。源码、manifest、80轮实际学习率、逐图均值和四次Train诊断状态恢复均核验通过，未访问Test。

RS2 Val IoU **72.8395%**、Dice **82.5258%**；比R2 IoU **+0.1741个百分点**，95%图像配对区间[-0.1181,+0.4575]，低于预设+0.3个百分点门且区间跨0。小面积四分位Dice/Recall亦下降，7项门仅通过3项。相对RS1 IoU **-0.2369个百分点**，区间[-0.5454,+0.0522]，未证明局部设计有独立增益。见[RS2报告](rs2_results/REPORT.md)与`independent_verification.json`。历史训练IoU与导出差4.0067e-8，在原核验界限内，没有追加Val推理或更改指标。

RS2已备份至HF Bucket `razaxq/BetterLViT/2a389922/`，20文件、1,694,460,839字节，远端及独立本地清单路径/大小/Xet全部一致，11项下载run产物也匹配。见`rs2_results/hf_upload_verified.json`、`download_xet_verified.json`。

RS1已备份的Last（845,347,135字节）已复制并核验后迁至`/root/regional_model_archive/b4dd566a/last_model-BetterLViT.pth.tar`，原路径保留符号链接，Best保留原位。迁移前审计无活跃文件/内存映射引用，复制后Xet/SHA256、检查点完整SHA/80轮/历史均核验。迁移前后训练盘可用**3,439,378,432 → 4,284,727,296字节**，系统盘可用**3,974,262,784 → 3,128,893,440字节**，共享fs保持**18,559,782,256字节**。没有丢弃模型、改动数据集或环境。见`storage_migration_rs1_last.json`；归档提交`4d2457182234e5fcad3afaa3533fd68c582bde43`已推送GitHub并独立核验，随后接续RS3。

### RS2原始提交与首检

RS2于悉尼**2026-09-11 11:59:42.365**执行首次短连接检查：已完整完成4/80轮，第5轮完成357/357训练批次并进入Val；GPU83%、显存17402MiB，源码/manifest一致，跟踪文件干净，日志尾部无异常。当前无RS2完整结果，未访问Test，检查预算已用1/2。第2—4轮平均216.155973秒，加120秒后续Train诊断余量，预测**今天16:32:12悉尼时间**训练结束。

同一heartbeat `betterlvit` 已改约**今天16:45悉尼时间唯一末检**，实际保存的rrule/目标thread/提示及预算均核验。之后不再中途连接；末检时核对距真实结束≤30分钟，不能把预测当成实际完成。见`rs2_first_snapshot.json`、`rs2_final_automation_verified.json`。

原始RS2提交为11:41:35.575，PID249881，SHA `2a389922cadb31a8bf7660bc33b3cb1ae94f20ab`，tag `experiment-rs2-regional-80e-seed1219-20260911`。固定80轮、seed1219、56窗口/28步长自然平均、λ=0.128312。原始11:57首检预约保留于`rs2_automation_verified.json`；本次实际检查11:59:42，后续以新的唯一末检预约为准。

RS1于悉尼2026-09-11 11:11:41.771完成80轮训练，11:13:25.457完成1429张Val导出；Best69，来源`b4dd566ae472079c55e41cffd7727060bcfd6bf5`。两项退出码0，完整SHA、实际80轮学习率、逐图指标、四次Train诊断和Best/Last已核验，未访问Test。11:28:34.230执行唯一末检，距训练结束1012.459秒（16.87分钟），符合≤30分钟，检查预算2/2已用完。

RS1 Val IoU **73.0764%**、Dice **82.7039%**；相对R2 IoU **+0.4111个百分点**，10000次图像配对95%区间[+0.1181,+0.7023]。Precision下降0.3710个百分点，7项门通过6项，整体未过门。只作单种子Val证据，不称稳定增益。详见[结果报告](rs1_results/REPORT.md)与`independent_verification.json`。

训练历史的>=0.5/float32与原逐图导出的>0.5/float64有继承差异。另一次有界的已完成Best Val诊断确认恰好1个背景像素等于0.5，完整解释IoU差1.135662367e-7，两套结果复算残差均为0，检查点SHA256前后相同；没有更改原结果、阈值、Best或冻结源码。见`rs1_results/iou_reconciliation.json`。该诊断不是额外训练状态检查。

RS1原始首检06:35、预测11:14:25结束、预约11:27末检的证据保留于`rs1_first_snapshot.json`、`rs1_final_automation_verified.json`，真实末检见`rs1_final_snapshot.json`。HF Bucket `razaxq/BetterLViT/b4dd566a/`已添加21文件、1,694,955,060字节，远端及独立本地清单的路径/大小/Xet哈希全部匹配；11项下载run产物字节/Xet亦匹配。Best/Last保留，fs仍18,559,782,256字节，训练盘备份后余5,155,680,256字节。见`rs1_results/hf_upload_verified.json`、`download_xet_verified.json`；归档提交`5ade2057a6d2b27e183834c43f0d4827cd8a4425`已推送GitHub并独立读取分支核验，随后接续RS2。

RS2、RS3已冻结部署、完成全部CUDA预检并推送GitHub；RS2已完成，RS3已提交。三组新损失关闭时与正式R2分组优化器五步逐值一致，各组插入Train诊断前后五步也逐值一致；11项loss行为和5项筛选门检查通过。完整核验见`preflight_verified.json`。原始校准和预检临时权重不会用于正式训练。

## 准备和冻结

共同系数为0.128312，epsilon=1e-6。先固定校准规则，再用32张固定Train图像、未训练R2的最终logit梯度计算：共同系数使12个批次×模式观察中的最大新增/主梯度范数比不超过10%。没有访问Val/Test或根据最终表现调节系数；此初始界限不代表整个训练过程。完整记录见`preflight/calibration.json`。

三组完整SHA、tag、远端目录和本地工作树见`sources.json`；GitHub分支与tag已独立核对，见`github_sources_verified.json`。每组manifest另存为`rsN_manifest.json`。正式启动必须通过`verify_preflight.py`：历史R2五步精确复现；每组新损失关闭后的正式分组优化器更新与R2相同；每组真实batch16前后向重复、插入诊断前后逐值一致。

11项loss行为检查及5项筛选门检查已实现。最初double精度测试夹具与原Focal强制float32不兼容，已把该项基线一致性检查改为生产float32输入并通过；原始失败记录保留。区域数值/梯度的独立裁窗参考检查仍使用double，没有改动主Focal实现。

## 运行、预约与后续接续

实际状态只能根据`rsN_launch.json`、`rsN_state.json`和最近预约/检查回执判断；有冻结代码或预检不代表训练已开始。首轮按以下阶段入口操作，不能一次把检查命令全部执行：

```powershell
python -X utf8 verify_preflight.py
python -X utf8 launch.py --label rs1
# 在提交回执 planned_first_check_sydney 对应时刻：
python -X utf8 inspect_run.py --label rs1 --phase first
# 用首次检查 forecast.final_check_sydney 更新同一个当前任务heartbeat；到时：
python -X utf8 inspect_run.py --label rs1 --phase final
python -X utf8 archive_completed.py --label rs1
# 若历史与导出IoU差超过原核验界限，先执行一次完成模型的Val指标诊断：
# python -X utf8 reconcile_iou.py --label rs1
python -X utf8 analyze_completed.py --label rs1
python -X utf8 upload_completed.py --label rs1
python -X utf8 verify_downloads.py --label rs1
# 完成记录、HF核验及文档提交推送后：
python -X utf8 launch.py --label rs2
```

RS2、RS3重复相同步骤。每次提交后将同一heartbeat改约新run的首检；首检约15分钟，依据已完成轮时预测结束，唯一末检约预计结束后12分钟。每组最多两次状态检查，必须核对真实结束至末检≤1800秒。不保持SSH、不循环轮询；若末检仍未完成，记录预测失准和预算耗尽，不能虚报符合时限或继续无上限检查。

训练程序在20/40/60/80轮内置32张Train的诊断，并恢复模型状态、参数梯度、模式和RNG；这不是远程状态查询。诊断梯度是最终logit空间，不是共享特征/参数梯度，不能与旧S2的20/64特征梯度观察混为一谈。每组成功训练后自动完整Val导出，此阶段不访问Test。

上一组完成、归档、独立复算和HF文件大小/Xet哈希验证后才提交下一组，不要求数值过门；RS1/RS2/RS3固定全部完成后才作阶段决策。原先三个run结束后删除heartbeat的收尾安排，已被用户新增文字研究授权更新为继续沿用同一预约接续A/B；不自动启动区域监督RS4、RS5或150轮。

RS3完成后的`archive_completed.py`还会调用`compare_grouping.py`，沿用冻结比较器的10000次图像配对bootstrap，输出`rs2_vs_rs3.json`；独立复算、报告及HF备份均包含该对照。按原计划只以IoU区间下界>0作为分组归因必要条件，不额外套用RS1区域增量的+0.001门，也不替代R2/RS1两项正式筛选。

## 存储与证据

初始训练盘可用6,879,092,736字节，共享fs18,559,782,256字节，GPU空闲。每次提交前重新要求目标盘可用>4,000,000,000字节、共享fs<20,000,000,000字节。三组保留Best/Last将新增约5.1GB，因此RS3接续前可能需要整理空间；不能绕过启动门，也不能因已上传就未经源文件、活跃引用与云端哈希核验直接删模型。优先处理可重建缓存，或保留可访问路径地迁移已完成、已备份的文件到有余量的存储，并记录实际字节和校验。

完整源码、校准、预检、manifest、runtime、日志、Val、历史、诊断、比较与检查预约都提交GitHub。正式训练结果按短SHA目录添加至HF Bucket `razaxq/BetterLViT`；上传使用现有认证缓存，不把凭据写入源码、命令行或日志。实际上传时遵循现有HF上传技能及核验流程。

研究计划是`709e4ecbe7d34d3ca01b6a48bdc6259b82026d10`中的`docs/REGIONAL_SUPERVISION_EXPERIMENT_PLAN_20260911.md`。实际运行manifest覆盖所有具体训练字段，但不改变该计划的阶段门和Test范围。

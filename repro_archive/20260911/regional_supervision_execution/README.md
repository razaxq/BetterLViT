# RS1 → RS2 → RS3执行入口

本轮用户于2026-09-11明确要求“启动训练”。固定执行三个首轮候选，各80轮、seed1219、batch16、R2单次余弦、原Dice/Focal。RS1整图软IoU，RS2最终输出的56窗口/28步长局部软IoU，RS3仅改变为阳性/背景分组。按[阶段计划](../../../docs/REGIONAL_SUPERVISION_EXPERIMENT_PLAN_20260911.md)推进，不按前一组数值修改后一组。

## 当前状态：RS1完成并备份，RS2已提交、首检已预约

RS2于悉尼**2026-09-11 11:41:35.575**提交后台训练，PID249881，SHA `2a389922cadb31a8bf7660bc33b3cb1ae94f20ab`，tag `experiment-rs2-regional-80e-seed1219-20260911`。固定80轮、seed1219、56窗口/28步长自然平均、λ=0.128312。当前为提交回执，尚未首检确认训练健康，无RS2完整结果，检查0/2。同一heartbeat `betterlvit` 已改约**今天11:57悉尼时间首次检查**，保存的rrule/目标thread/提示及预算均核验。届时用实测轮时预约唯一末检；不得现在额外连接。见`rs2_launch.json`、`rs2_automation_verified.json`。

RS1于悉尼2026-09-11 11:11:41.771完成80轮训练，11:13:25.457完成1429张Val导出；Best69，来源`b4dd566ae472079c55e41cffd7727060bcfd6bf5`。两项退出码0，完整SHA、实际80轮学习率、逐图指标、四次Train诊断和Best/Last已核验，未访问Test。11:28:34.230执行唯一末检，距训练结束1012.459秒（16.87分钟），符合≤30分钟，检查预算2/2已用完。

RS1 Val IoU **73.0764%**、Dice **82.7039%**；相对R2 IoU **+0.4111个百分点**，10000次图像配对95%区间[+0.1181,+0.7023]。Precision下降0.3710个百分点，7项门通过6项，整体未过门。只作单种子Val证据，不称稳定增益。详见[结果报告](rs1_results/REPORT.md)与`independent_verification.json`。

训练历史的>=0.5/float32与原逐图导出的>0.5/float64有继承差异。另一次有界的已完成Best Val诊断确认恰好1个背景像素等于0.5，完整解释IoU差1.135662367e-7，两套结果复算残差均为0，检查点SHA256前后相同；没有更改原结果、阈值、Best或冻结源码。见`rs1_results/iou_reconciliation.json`。该诊断不是额外训练状态检查。

RS1原始首检06:35、预测11:14:25结束、预约11:27末检的证据保留于`rs1_first_snapshot.json`、`rs1_final_automation_verified.json`，真实末检见`rs1_final_snapshot.json`。HF Bucket `razaxq/BetterLViT/b4dd566a/`已添加21文件、1,694,955,060字节，远端及独立本地清单的路径/大小/Xet哈希全部匹配；11项下载run产物字节/Xet亦匹配。Best/Last保留，fs仍18,559,782,256字节，训练盘备份后余5,155,680,256字节。见`rs1_results/hf_upload_verified.json`、`download_xet_verified.json`；归档提交`5ade2057a6d2b27e183834c43f0d4827cd8a4425`已推送GitHub并独立读取分支核验，随后接续RS2。

RS2、RS3已冻结部署、完成全部CUDA预检并推送GitHub；RS2已提交，RS3尚未提交。三组新损失关闭时与正式R2分组优化器五步逐值一致，各组插入Train诊断前后五步也逐值一致；11项loss行为和5项筛选门检查通过。完整核验见`preflight_verified.json`。原始校准和预检临时权重不会用于正式训练。

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

上一组完成、归档、独立复算和HF文件大小/Xet哈希验证后才提交下一组，不要求数值过门；RS1/RS2/RS3固定全部完成后才作阶段决策。三个run结束后删除当前heartbeat，不自动启动RS4、RS5或150轮。

## 存储与证据

初始训练盘可用6,879,092,736字节，共享fs18,559,782,256字节，GPU空闲。每次提交前重新要求目标盘可用>4,000,000,000字节、共享fs<20,000,000,000字节。三组保留Best/Last将新增约5.1GB，因此RS3接续前可能需要整理空间；不能绕过启动门，也不能因已上传就未经源文件、活跃引用与云端哈希核验直接删模型。优先处理可重建缓存，或保留可访问路径地迁移已完成、已备份的文件到有余量的存储，并记录实际字节和校验。

完整源码、校准、预检、manifest、runtime、日志、Val、历史、诊断、比较与检查预约都提交GitHub。正式训练结果按短SHA目录添加至HF Bucket `razaxq/BetterLViT`；上传使用现有认证缓存，不把凭据写入源码、命令行或日志。实际上传时遵循现有HF上传技能及核验流程。

研究计划是`709e4ecbe7d34d3ca01b6a48bdc6259b82026d10`中的`docs/REGIONAL_SUPERVISION_EXPERIMENT_PLAN_20260911.md`。实际运行manifest覆盖所有具体训练字段，但不改变该计划的阶段门和Test范围。

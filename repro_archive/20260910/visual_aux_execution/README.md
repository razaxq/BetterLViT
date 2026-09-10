# R2视觉辅助监督实验执行入口

以[冻结计划](PLAN.md)为准。R2作为历史基线，S1仅像素监督，S2像素/区域存在/占比监督；两组80轮单次余弦、从头训练、seed1219。先S1，完成后归档再S2。配置不因中间结果改变。

当前文件用于后续定时继续执行。实际是否已启动、检查次数和预约，以各自`*_launch.json`、`*_state.json`与最近一次`*_automation*.json`回执为准；没有这些文件不能宣称启动。完整SHA/tag/远端路径见[sources.json](sources.json)。

## S1首次预约检查

更新：S1已完成，以下首检记录为历史阶段；当前完成状态见下节。

2026-09-10 17:25:51.738（悉尼），heartbeat按预约触发，执行唯一一次短连接。已完成4/80轮，正在第5轮；GPU93%、显存17,840MiB，运行源码与冻结SHA一致且跟踪文件干净。日志尾部正常推进，当前没有完整结果或Test评估。首次五轮排除Best选择是原R2共同规则，日志仍显示best_epoch=1为初始占位，不能解读为已有正式Best。[原始首检快照](s1_first_snapshot.json)。

第2–4轮平均219.600946秒；按最新完整轮结束时间外推剩余76轮，并加入120秒四次Train诊断余量，预计训练在今天22:04:23.275结束。同一heartbeat `betterlvit` 已改约今天22:17唯一末检，app保存配置核验通过。实际结束时间和30分钟间隔需末检确认，当前检查预算1/2，不再中途连接。S2继续等待S1完成归档。[预测与当前状态](s1_state.json)、[末检预约回执](s1_final_automation.json)、[配置核验](s1_final_automation_verified.json)。

## S1完成与判定

S1在2026-09-10 22:13:02.833（悉尼）完成80轮，22:14:48.068完成完整1429张Val导出；22:19:31.670末检，距训练结束388.837秒，预算2/2。原预约22:17，本次自动触发约晚2分钟；实际结束比首检预测晚519.558秒，仍满足结束后30分钟检查要求。GPU空闲、来源一致且跟踪文件干净，训练/评估返回码均0。

Best为epoch67。Val macro IoU72.9003%、Dice82.6077%；对R2分别+0.2350、+0.2043个百分点。IoU区间[-0.0330,+0.4986]个百分点跨0且低于+0.3门槛，因此筛选未通过，不扩展S1多种子或Test。其他5项非退化条件均通过。S2仍按冻结计划继续。[完整报告](s1_results/REPORT.md)、[独立核验](s1_results/independent_verification.json)。

HF正在添加19文件、1,698,048,537逻辑字节至`1f7edb78/`，包含原Best/Last、完整训练源码、manifest、日志、Val、80轮历史、四次诊断及对照分析。是否完成以`hf_upload_verified.json`为准；本段不代替上传核验。

## 本次提交

S1已于2026-09-10 17:09:02.700（悉尼）提交后台，PID208088，检查预算0/2；这是提交回执，尚未进行首次健康检查。S2源码已冻结，等待S1完成归档后启动。启动时训练盘可用10,322,718,720字节，共享fs18,559,782,256字节。[S1回执](s1_launch.json)。

当前任务heartbeat `betterlvit` 已创建为ACTIVE，首检预约2026-09-10 17:25（悉尼），与S1回执及app保存配置核对一致；首检之后才根据实测轮时确定末检。定时规则是需要随阶段更新的每日单时刻预约，不能保留旧时间等待次日再检查。[创建回执](automation.json)、[配置核验](automation_verified.json)。

## 操作顺序

在本目录运行：

```powershell
python -X utf8 verify_preflight.py
python -X utf8 launch.py --label s1
python -X utf8 inspect_run.py --label s1 --phase first
python -X utf8 inspect_run.py --label s1 --phase final
python -X utf8 archive_completed.py --label s1
python -X utf8 analyze_completed.py --label s1
python -X utf8 upload_completed.py --label s1
python -X utf8 launch.py --label s2
```

S2使用相同first/final/归档流程。上面的命令是阶段入口，**不能一次连续全部运行**：按预约每次只做当前到期步骤。launch只提交并断开；首检根据实测轮时生成末检时间，需用Codex automation工具把同一个当前任务heartbeat更新到预测时刻。首次与最终检查均计入预算，每组最多两次。不要用持续连接、睡眠循环或额外轮询确认训练状态。

若首检已经发现完成，记录完成间隔后归档，不再做多余final检查。若实际失败，保留证据并诊断，不能当成数值负结果。若末检仍未结束，记录预测失准和用完的预算，不虚称满足“结束后30分钟”，不要无视预算反复连接。

## 完成后的必要动作

归档脚本只在已取得complete快照后读取静态结果，核对完整SHA、80轮历史、Best/Last、四次Train诊断与1429张Val。自动输出R2对照门；S2还输出相对S1的区域增量门。所有条件、失败项和原始数值都报告；这是Val筛选，不是稳定Test结果。

每组完成后按用户授权上传HF Bucket `razaxq/BetterLViT`，按训练Git短SHA隔离，包含Best/Last、日志、manifest、四次Train遥测及Val结果。可读取已有 `betterlvit-hf-bucket-upload` 技能及本项目最新上传脚本，遵循本次明确授权范围；验证源文件大小与Xet哈希后记录回执。原始模型保留，不能根据“上传命令成功”就删除本地模型。

将源代码、执行/结果文档和实验台账全部提交推送 `https://github.com/razaxq/BetterLViT.git`。文档工作树分支 `docs/experiment-tracker`；origin为本机bare仓库，推送应明确使用GitHub URL。不要修改已启动的训练来源或复用标签覆盖旧结果。

S1数值无论好坏，都完成已授权的S2配置。两组完成后依PLAN决定匹配种子复验或停止；普通深监督收益不自动等同第二创新。所有工作结束后删除当前heartbeat；状态无变化时不发送重复通知。

## 已完成的准备

已清理8个可重建视觉探针NPY缓存，逻辑字节6,336,072,704。训练盘可用从3,990,876,160到10,326,982,656字节；共享fs清理前后均18,559,782,256字节。模型、原数据、探针结果与代码保留。[清理证据](storage_cleanup.json)。

基线兼容、监督行为、判定门与CUDA预检原始结果在`preflight/`。`preflight_verified.json`存在且status为ok才满足完整启动前置条件。每个候选在最终源码上做一次正常五步及一次插入Train诊断的五步，必须逐步输出/loss一致；S2关闭辅助loss还须和正式优化器的R2五步完全一致。

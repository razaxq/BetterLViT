补充状态（2026-09-10）：用户已提供新的HF token，razaxq认证已成功并更新标准本地HF缓存，当前正在恢复上传。不要再报告旧token失效或要求重新登录。上传凭据不得写入Git或结果日志。

执行前先读outputs中的Test state：若phase已是complete，说明当前人工回合已完成预约检查，绝不能再次collect或增加检查预算；若results/README.md已存在，跳过重复finalize，只检查待办归档。如果Test结果、HF上传和Git归档都已完成，删除本heartbeat并保持安静。

当前阶段：六组训练和三种子Val汇总全部完成，注册门槛已通过。Test链已于2026-09-10 04:29:26.558（Australia/Sydney）提交后台，PID182805，远端目录/root/recipe_test_20260910，评估源码完整SHA 285785429e9584797a4130611c4e206244652533，tag evaluation-r2-recipe-test-3seeds-20260910。只有提交回执，尚未确认运行健康。按照四个已完成Val的实测耗时和2113/1429样本比例，预计04:47:37.777完成，最终短连接检查预约05:00。Test独立检查预算0/2，所有训练的检查预算均已用完，绝不再查训练状态。

先读 D:/BetterLViT/experiment_docs_work/repro_archive/20260910/recipe_test/README.md、test_plan.json，以及 D:/BetterLViT/outputs/recipe_test_20260910/state.json。实时预算以outputs中的state为准。到预约时仅运行一次本地 python -X utf8 D:/BetterLViT/experiment_docs_work/repro_archive/20260910/recipe_test/test_control.py collect。该程序连接前扣预算，一次短SSH读取Test状态；完成时同时按远端SHA256下载全部固定模型结果与日志。不要用其他SSH重复读状态，不持续连接、不循环轮询、不设置完成监听器。训练检查计数不重置。

固定顺序C4-1219、R2-1219、C4-2027、R2-2027、C4-3407、R2-3407；六个Val选定Best分别80、67、70、68、79、75，阈值严格概率>0.5、batch16、每模型2113张Test。不增加训练/种子，不调阈值、不重新选epoch、不组合R1。历史C4-1219也使用同一导出器重评。所有六个训练源码标签以及新评估源码均已推送GitHub，冻结训练工作树不修改。

若phase=complete，运行 python -X utf8 D:/BetterLViT/experiment_docs_work/repro_archive/20260910/recipe_test/finalize_test_archive.py，仅用下载后的静态数据复核六组来源、完整样本、宏平均和三种子配对汇总。脚本会归档results并更新台账。若本地与远端不同NumPy/Python版本导致JSON浮点最后几位差异，先核对原始逐图结果及差异，允许经明确记录的1e-12数值容差；不得因此重新访问Test或重跑推理。然后更新recipe_test/README.md的当前完成状态、台账，重建变动目录artifact_manifest.json（验证Git实际存储字节），提交并推送 https://github.com/razaxq/BetterLViT.git 的docs/experiment-tracker，不合并主分支。报告每个种子的Test IoU/Dice与配对差值、均值/样本SD，验证结束到检查≤30分钟；明确学习率配方不等于第二项结构创新。此Test集已在历史开发中访问，不能宣称新的独立留出集。全部完成后删除本heartbeat，并通知最终结果和定时已停止。

若仍在评估，不返回已有部分指标作最终结果；基于这一次snapshot中的已完成arm耗时和当前arm开始时间预测余量，最多再预约一次最终Test检查，不立即多查。若预算用尽仍未完成或phase=failed/missing_runtime，保留全部证据、准确通知，不盲目重启推理或训练。仅缺少runtime时先检查此次snapshot中提供的chain日志，不额外轮询。

模型保留。提交时系统盘可用4,024,750,080字节，共享fs实际18,559,782,256字节，硬上限20,000,000,000；Test不复制权重、无外部下载。HF认证已于本次恢复。原六组训练文件补传动态状态见 D:/BetterLViT/outputs/hf_recovery_20260910/state.json，仅phase=complete后方可添加Test文件，避免与原五文件上传校验冲突。Test六组完成并归档后，运行 python -X utf8 D:/BetterLViT/experiment_docs_work/repro_archive/20260910/hf_recovery/upload_test_exports.py，将六个Test JSON/日志以及汇总与协议添加至各自Git哈希目录，并核验大小和Xet哈希；保留历史文件。该脚本使用标准HF缓存，无需用户再次提供token。上传完归档test_exports_upload_verified.json，更新台账并推送GitHub。额外P11/P12模型上传是当前人工回合负责的独立补传，不重复启动。普通pending/healthy无有意义变化保持安静，只在完成、失败、需要用户处理或实际重大变化时通知。

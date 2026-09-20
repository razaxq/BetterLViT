# PLAM 基线核查（2026-09-16）

## 结论

J0 保留原版 PLAM 解码模块，但整套模型不是官方 LViT 的原样复现。应称为“LViT-PLAM（冻结 CXR-BERT、32-token 文本、统一训练设置）”。当前证据未发现 FSDR/RACE/LoRA 残留或结果均值计算错误；同时确认训练/验证划分存在已知患者标识重叠，不能声称完整患者独立。

## 分数与配置

官方仓库 README 的 QaTa-COV19 LViT-T：Dice 83.66%，IoU 75.11%。当前 J0 三种子 Test 均值：Dice 84.0251%，IoU 75.4775%；相差分别为 +0.3651 和 +0.3675 个百分点，属于跨实现参考差异，不是模块贡献。

官方当前公开源码使用 BertEmbedding，文本截断到 10 个 token；当前模型使用冻结 CXR-BERT、32 个 token，无 LoRA。官方公开训练代码使用 Dice/BCE、余弦热重启、Val Dice 选权重；当前使用 Dice/Focal、单周期余弦 3e-4→1e-6、Adam weight decay=1e-4、80 轮、batch 16、Val IoU 选权重。公开源码默认任务为 MoNuSeg，不能将其所有默认参数当成论文 QaTa 的实际运行配置。

这些差异可能影响成绩，但此次核查没有分离各项差异的因果贡献。官方测试代码与本地都采用逐图 Dice/IoU 后求均值、阈值 0.5，不能将差异简单归因于 macro/micro 混用。

## 权重与数据检查

- 三个 J0 Best 权重 SHA256 与实验登记一致；每个权重含 32 个 PLAM 参数/缓冲张量，没有 FSDR、RACE、LoRA 参数张量。元数据均为 legacy_plam，RACE 关闭。
- PLAM PixLevelModule 类的 AST 与下载的官方代码一致。匹配初始化重放仅使用随机初始化的临时模块，不加载已训练 FSDR 权重。
- 已有汇总核对全部 12 个模型的逐图 Dice/IoU 与均值，以及同组评估图像集合、阈值、Best 权重来源。
- 服务器物理数据目录为 Train 5716、Val 1429、Test 2113；图像与掩膜名称配对正确。跨划分文件名、文件字节哈希和解码像素哈希完全重复数量均为零。
- 可恢复的 sub-S… 患者标识在 Train/Val 间重叠 434 个；Train/Test 与 Val/Test 间这类标识重叠均为零。没有患者标识的图像不在患者级核查覆盖范围；此检查也不排除近重复图像。
- J0 训练源码直接使用 Train_Folder/Val_Folder，并未使用另一个工作树已有的 known_patient_grouped_sensitivity_v1 清单。历史 RESEARCH_PROTOCOL_RESET_2026-08-16.md 已记录相同的 434 个重叠；此次重新从服务器确认，不能作为新发现的分数抬升因果证据。
- 患者跨 Train/Val 的重叠可能影响验证评估与 Best 选择；这不等同于已证实 Test 数据参与训练。官方论文划分与当前目录尚未逐样本核对，因此不声称与官方完全同协议。

## 论文报告边界

现有统一设置下，FSDR 相对 J0 的平均 Test IoU 增量为 +0.5058 个百分点；完整 RACE 加入 FSDR 为 +0.2428 个百分点；两模块整体相对 J0 为 +0.7487 个百分点。不得用完整模型与官方 75.11% 的全部差值替代模块消融增量。

现有结果保留为当前划分下的匹配消融证据。若需更严格的患者泛化结论，应对四组共同执行已知患者分组的敏感性实验并披露匿名样本限制；若需官方严格复现，应单独对齐文本编码、训练设置、源码版本与数据清单。本次未启动训练、未修改权重或已有评估分数。

## 证据

- 实时服务器核查：server_audit.json；执行脚本：check_server.py。
- 汇总：../stage1_overall_20260915/FINAL_REPORT.md、FINAL_RESULTS.json。
- 本地源码：../../stage1_j0s1219_work/train_model.py、Config.py、nets/LViT.py、nets/Vit.py、nets/BetterLViT.py。
- 既有划分记录：../../BetterLViT-Migrated/docs/RESEARCH_PROTOCOL_RESET_2026-08-16.md。
- 官方来源：https://github.com/HUANGLIZI/LViT 。此次下载的当前 main 源码保存在 official/；未声称该版本是论文实际运行的精确提交。

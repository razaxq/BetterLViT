# S2 Val筛选结果

完成80轮，Val IoU选定epoch 75。来源 `7defc637e36974fabd0f47763275c90f617e5f53`；tag `experiment-s2-r2-regional-80e-seed1219-20260910`。

| 配置 | Val IoU | Dice | Precision | Recall | Brier |
|---|---:|---:|---:|---:|---:|
| R2 | 72.6654% | 82.4034% | 80.5086% | 88.0502% | 0.021162 |
| S1 | 72.9003% | 82.6077% | 80.5860% | 88.3938% | 0.021017 |
| S2 | 72.7415% | 82.4575% | 81.2988% | 87.4212% | 0.020642 |

相对R2：IoU +0.0761个百分点，95%图像配对区间[-0.2129, +0.3642]；Dice +0.0541个百分点。筛选通过：False；失败条件：iou_minimum, iou_ci_positive, small_dice_not_lower, small_recall_not_lower。

总体Dice/precision、最小GT总面积组Dice/recall和Brier条件见[r2_vs_s2.json](r2_vs_s2.json)。1429张逐图均值、差值、80轮LR、Best选择与四次固定32张Train诊断独立核验通过。Train梯度为四批eval快照均值，不能直接归因为IoU贡献。

训练结束2026-09-11 03:36:20，Val完成2026-09-11 03:38:03，末检2026-09-11 03:48:11（悉尼）；末检距训练结束710.834秒，检查2/2。

当前结果全部为Val，尚无本模型Test成绩。只有通过注册R2门的候选才进入匹配种子复验；区域创新价值还需通过S2对S1的增量门。HF上传完成与否另见上传核验回执。

相对S1：IoU -0.1588个百分点，95%图像配对区间[-0.4633, +0.1433]；区域增量门通过：False。失败条件：iou_minimum, iou_ci_positive, dice_not_lower, small_dice_not_lower, small_recall_not_lower。

这项比较同时加入存在与占比任务，不能区分两者各自作用。未过门则按计划停止当前配置，不延长150轮、不调系数，也不能称第二创新已成功。

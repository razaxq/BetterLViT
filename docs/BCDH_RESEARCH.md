# BCDH 深入研究：无边界损失的双头校正解码器

更新时间：2026-09-02（Australia/Sydney）

## 结论先行

BCDH 可以作为 BetterLViT 第二项架构创新继续开发，但必须重新定义：它不应是“分割头 + 边界头”，而应是 **两个都预测完整病灶掩码的跨尺度分割头，加上由预测本身产生的不确定性与有符号分歧提示，驱动一个零初始化、幅度受限的残差校正器**。

推荐暂名：**BCDH-R（Boundary-Conscious Dual-Head Refiner，边界感知双头校正器）**。不建议现阶段把 C 解释为 *Calibrated*；除非后续 Brier score/ECE 确实改善，否则“校准”会被审稿人理解为概率校准主张。

该设计满足当前硬约束：

- `Frozen CXR-BERT`，`LoRA=False`；
- 保留已验证的 `FAM-EPPA V4-B`；
- 主损失与辅助损失均为 `0.5 Dice + 0.5 Focal`，Focal `gamma=2`；
- `boundary_loss=0.0`；
- 不生成 GT 边界图、距离变换、方向图、Sobel/Canny 标签，也不使用 surface、Hausdorff 或 boundary Dice loss；
- 首轮只做 40 epoch validation-only 配对，不访问 Test。

我的判断是：这是当前比继续修补 TCSR 更有研究价值的方向。它直接处理“预测哪里可能错、应向内收缩还是向外补偿”，而不是继续改写 skip feature；因此与 EPPA 的特征融合职责更正交。

## 1. BCDH 要解决的实际问题

现有证据显示 TCSR V1–V2.5 的主要失败不是 gate 完全失效，而是路由对最终误差缺少直接约束：即使 gate 范围、稀疏度、identity 和训练稳定性正常，模型仍容易扩大预测区域并损失 precision。P5 加入边界监督后仍未超过严格对照，而且最终方法又禁止 boundary loss。

BCDH 因此不再回答“哪些 skip feature 应通过”，而回答两个更靠近输出的问题：

1. 粗尺度语义头和细尺度主头在哪些像素上不确定或意见不同？
2. 在不接触边界标签的前提下，最终输出应在这些位置做小幅正向还是负向修正？

这两个问题分别对应跨尺度预测和有符号残差校正。

## 2. 文献证据与可用边界

### 2.1 可以借鉴的机制

| 工作 | 原始机制 | 对 BCDH 的启发 | 不能直接照搬的部分 |
|---|---|---|---|
| [BASNet（CVPR 2019）](https://openaccess.thecvf.com/content_CVPR_2019/html/Qin_BASNet_Boundary-Aware_Salient_Object_Detection_CVPR_2019_paper.html) | 先预测粗图，再用残差模块细化；所有 side output 仍与完整 mask 比较 | “预测—修正”可以在没有单独边界标签的情况下改善细结构 | 原模型 refiner 很重，且使用 BCE+SSIM+IoU；本项目不应同时改变架构和主损失 |
| [PointRend（CVPR 2020）](https://arxiv.org/abs/1912.08193) | 根据当前预测的不确定性，优先细化模糊位置 | 不确定性可以由预测概率本身得到，无需边界 target | 稀疏点采样/`grid_sample` 增加实现与确定性 CUDA 风险；本项目改用稠密提示图 |
| [UNet++（DLMIA 2018）](https://arxiv.org/abs/1807.10165) | 多尺度 full-mask side output 和 deep supervision | 112×112 粗头可直接使用原始 mask 监督 | 单纯“加辅助头”已有大量先例，不能独立构成核心创新 |
| [UNet 3+（ICASSP 2020）](https://arxiv.org/abs/2004.08790) | 全尺度连接和多级深监督 | 支持跨尺度完整分割头的训练合理性 | 其 Focal+MS-SSIM+IoU 会引入额外损失变量，不适合作为首轮配对 |
| [PraNet（MICCAI 2020）](https://arxiv.org/abs/2006.11392) | 反向注意逐级挖掘当前预测遗漏区域，边界效果是隐式出现的 | 预测的“支持/反对”信息可用于定向修正 | 官方训练常用 GT 邻域加权的 structure loss；这接近边界加权监督，本项目不采用 |

BASNet 是最接近本项目约束的架构依据：其核心是粗预测后的残差 refinement，而不是必须拥有边界标签。PointRend 则证明预测不确定性是定位精细修正区域的有效信号。两者结合后，适合形成“预测产生提示、稠密小残差校正”的 BetterLViT 版本。

### 2.2 明确排除的方法

| 工作 | 排除原因 |
|---|---|
| [Gated-SCNN（ICCV 2019）](https://openaccess.thecvf.com/content_ICCV_2019/html/Takikawa_Gated-SCNN_Gated_Shape_CNNs_for_Semantic_Segmentation_ICCV_2019_paper.html) | shape stream 使用边界监督；论文也明确指出 boundary loss 需要精细边界标注 |
| [SegFix（ECCV 2020）](https://arxiv.org/abs/2007.04269) | 从 GT mask 生成距离图、边界图和方向图，违反当前禁止项 |
| [Boundary Loss（MIDL 2019）](https://proceedings.mlr.press/v102/kervadec19a.html) | 明确以轮廓/距离度量构造目标，正是本项目禁止的监督形式 |
| P5/TCSR V2.5 | 使用 GT 边界 Dice 对齐与边界外泄漏项，只能保留为历史诊断证据 |

判断标准不是损失函数名字里是否含有 `boundary`，而是训练目标是否从 GT mask 派生出轮廓、距离、方向或窄带权重。只要有这种派生监督，就不进入最终方法。

## 3. 推荐的 BCDH-R V1

### 3.1 两个完整分割头

现有 A9/FAM-EPPA 解码器具有天然接入点：

- `up2` 后：`D2 ∈ R^(64×112×112)`，接轻量粗分割头；
- `up1` 后：`D1 ∈ R^(64×224×224)`，保留现有主分割头。

粗头不是边界头。它与主头一样预测完整病灶掩码，并只接受原始 segmentation mask 的 Dice/Focal 监督。

令：

```text
z_c = Upsample(Conv1x1(D2))
p_c = sigmoid(z_c)

z_b = Conv1x1(D1)
p_b = sigmoid(z_b)
```

`z_c/p_c` 是粗尺度语义预测，`z_b/p_b` 是未经 BCDH 修正的细尺度基线预测。

### 3.2 预测自身生成的四类提示

所有提示只依赖模型输出，并在送入 refiner 前 `stop-gradient`：

```text
uncertainty  u   = 4 * p_c * (1 - p_c)
fine-only    h_f = p_b * (1 - p_c)
coarse-only  h_c = p_c * (1 - p_b)
coarse prior     = p_c
```

- `u` 在粗头概率接近 0.5 时最大，提示模型“这里不确定”；
- `h_f` 表示细头认为是前景、粗头不支持的区域，是潜在孤立假阳性/过扩张候选；
- `h_c` 表示粗头支持但细头漏掉的区域，是潜在漏检候选；
- `p_c` 提供低分辨率整体语义。

`h_f/h_c` 只是预测分歧，不是真实 FP/FN 标签。它们不接触 GT，也不能被描述成边界监督。

使用 `stop-gradient` 是为了防止主头或粗头“操纵提示图”来降低最终损失；两个头仍分别通过正常分割损失学习。

### 3.3 稠密、受限的残差校正

将细尺度 feature 与四类提示拼接：

```text
r = R([D1, p_c, u, h_f, h_c])
delta = tanh(r)
z_final = z_b + delta_max * delta
p_final = sigmoid(z_final)
```

推荐 `R` 使用两层轻量 3×3 卷积（可采用 depthwise + pointwise）和一个 1×1 输出层；最后 1×1 卷积严格零初始化。`delta_max` 首轮固定为 `1.0 logit`，不另加可学习 gate。

这带来三个关键性质：

1. **初始化严格等价基线**：初始 `delta=0`，所以 `p_final=p_b`；
2. **不会重写整个解码器**：BCDH 只能在输出 logits 上做有界修正；
3. **正负修正均允许**：能够抑制过扩张，也能够恢复粗尺度支持的漏检区域。

不采用 PointRend 式稀疏采样。稠密提示仅增加很小的卷积开销，也避免 `grid_sample` 在确定性 CUDA 路径上的额外风险。

## 4. 损失函数：保持单一变量

V1 只使用完整 mask 的 Dice/Focal：

```text
L_seg(p, y) = 0.5 * L_Dice(p, y) + 0.5 * L_Focal(p, y)

L_total = L_seg(p_final, y) + 0.2 * L_seg(p_c, y)
```

固定：

- Focal `gamma=2`；
- 前景/背景分别求均值后按 `0.5/0.5` 合并；
- auxiliary weight `λ_aux=0.2`；
- `boundary_loss=0.0`。

V1 不加入 head-consistency loss。原因是分歧本身就是 refiner 的信息来源，过早强迫两头一致可能使双头退化为重复预测。如果观察到粗头长期失真，再把一致性作为独立消融，而不是默认组件。

## 5. 为什么它比现有候选更合理

| 候选 | 优点 | 关键问题 | 结论 |
|---|---|---|---|
| 显式 edge head | 论文叙述直观 | 没有边界 target 时难以定义独立任务；加 target 又违反约束 | 排除 |
| BASNet 式完整 U-Net refiner | 有成熟先例 | 参数重、与当前解码器重复、容易改变太多变量 | 不作为 V1 |
| PointRend 式不确定点采样 | 只修正困难像素 | 实现复杂、需要采样算子、确定性风险更高 | 只借鉴 uncertainty 思想 |
| 单纯 coarse auxiliary head | 简单、稳定 | 深监督是已知技术，创新性不足 | 只作为消融 |
| **BCDH-R V1** | 双尺度语义、预测不确定性、有符号修正、严格 identity、无边界监督 | 仍需实验证明增益与 boundary/calibration 指标 | **推荐** |

TCSR 在 feature routing 层与 FAM-EPPA 存在职责重叠；BCDH 则位于 output refinement 层。论文组合可清晰表述为：

1. **FAM-EPPA**：频率感知的多模态解码特征融合；
2. **BCDH-R**：无需边界监督的跨尺度预测误差校正。

两项创新分别解决 feature fusion 与 output correction，逻辑上更独立。

## 6. 与当前代码的具体适配

现有实现已经给出所需张量：

- 解码器各阶段与输出头位于 `nets/LViT.py`；`up2` 和 `up1` 当前连续覆盖同一个变量，因此实现时应保存 `d2`、`d1` 两个显式变量；
- 当前 `outc` 先经过 sigmoid 再返回概率；BCDH 应先保留 `base_logits`，相加残差后只做一次 sigmoid；
- 当前训练循环假定模型只返回一个 tensor；建议增加 `return_aux=False` 参数。普通 `forward` 和评估脚本仍返回最终 tensor，训练时使用 `return_aux=True` 返回 `final/coarse/base`；
- 训练 Dice/IoU、可视化、阈值选择和 Test JSON 只使用 `final`；`coarse/base` 仅用于辅助损失和机制诊断；
- 检查点必须记录 BCDH 版本、`lambda_aux`、`delta_max`、是否 stop-gradient、模块参数量及 `source_git_commit`。

这比全局改变模型返回协议更安全，也保持旧评估与历史检查点兼容。

## 7. 需要预注册的诊断指标

BCDH 不能只看最终 Dice；否则无法证明“第二项创新”的作用机制。首轮至少保存：

### 7.1 主性能

- validation macro Dice / IoU / precision / recall，阈值 0.5；
- 验证集选定阈值下同组指标，作为次要结果；
- 最小病灶面积四分位的 Dice 与 precision。

### 7.2 轮廓与概率质量（只作评估，不参与训练）

- boundary F1，容差固定为 2 pixels，并预注册空 mask 处理规则；
- Brier score；
- 如要保留 *calibrated* 命名，再加固定 bin 规则的 ECE。

用 GT 计算评估指标不等于边界监督；关键是它们不能进入反向传播或样本权重。

### 7.3 机制诊断

- `coarse/base/final` 三者各自 Dice、IoU、precision；
- `|delta|` 均值、95 分位数、正/负修正比例；
- 不确定度 top 20% 区域与其余区域的 residual 能量比；
- `h_f` 与 `h_c` 面积占比；
- base→final 改善像素与恶化像素比例；
- 初始化时 `max_abs(p_final - p_base) == 0`，重复前向与重复训练 batch 的确定性误差为 0。

## 8. 实验计划与阶段门

### 阶段 0：实现与确定性预检

必须通过：

- BCDH 关闭时与 A9 模型严格等价；
- BCDH 开启且零初始化时 `final == base`；
- 两次 batch16 前向/反向结果一致；
- no LoRA、`boundary_loss=0.0`；
- 所有 BCDH 参数有有限梯度；
- 峰值显存记录并低于服务器安全线。

### 阶段 1：严格配对筛选（不访问 Test）

| ID（建议） | 配置 | 训练 |
|---|---|---|
| C1 | Frozen CXR-BERT + FAM-EPPA V4-B + Dice/Focal + 无 BCDH | 40 epochs，batch16，seed1219，validation-only |
| P6 | 与 C1 完全相同 + BCDH-R V1 | 40 epochs，batch16，seed1219，validation-only |

P6 必须同时满足：

1. validation macro Dice 相对 C1 至少 `+0.002`；
2. macro precision 不下降；
3. 最小病灶四分位 Dice 与 precision 均不下降；
4. boundary F1 提升；
5. Brier score 不恶化；
6. 机制统计无 residual 饱和、全图修正或单向塌缩；
7. 无异常 train-validation gap。

任一关键门失败：停止，不访问 Test，不通过调阈值掩盖失败。

### 阶段 2：组件消融

仅 P6 通过后运行：

- auxiliary-only：只加粗头监督，不使用 refiner；
- uncertainty-only：refiner 只接 `p_c + u`；
- full signed disagreement：加入 `h_f + h_c`；
- 可选 no-stop-gradient，用于验证梯度隔离是否必要。

该阶段回答增益究竟来自普通 deep supervision、uncertainty，还是有符号跨尺度校正。

### 阶段 3：稳健性与正式实验

- 先增加训练 seed，验证均值、方差与 paired bootstrap；
- 再扩到 80 epochs validation-only；
- 只有多 seed 仍通过，才建立独立 Git 提交/标签并进行 150 epochs + 一次正式 Test；
- Test 主口径仍是逐样本 macro Dice/IoU，不切换到 FMISeg micro 口径。

## 9. 可以写进论文的论点，以及现在还不能写的内容

如果实验通过，可以主张：

> We introduce a boundary-conscious dual-head refiner that derives dense uncertainty and signed cross-scale disagreement cues solely from full-mask predictions, and applies a zero-initialized bounded logit residual to correct ambiguous regions without boundary annotations or boundary-specific losses.

中文对应：

> 提出一种无需边界标注与边界专用损失的边界感知双头校正器；该模块从粗、细完整分割预测中构造稠密不确定性与有符号跨尺度分歧提示，并通过零初始化、有界的 logit 残差对模糊区域进行校正。

现在不能提前宣称：

- “显著改善边界”——必须先有 boundary F1 证据；
- “概率校准更好”——必须先有 Brier/ECE 证据；
- “双头结构本身是创新”——deep supervision 和双头已有大量先例；
- “无需边界信息”——准确说法是无需**边界标签或边界损失**，模型仍利用预测不确定性关注边缘样区域；
- “BCDH 已经是第二项核心创新”——目前仍是研究设计，只有严格配对、多 seed 和消融通过后才成立。

## 10. 最终建议

下一步应开发 **BCDH-R V1**，但不要立刻跑 150 epochs。先建立 C1/P6 的 40-epoch validation-only 严格配对。实现时保持 A9 的 Frozen CXR-BERT、FAM-EPPA V4-B、Dice/Focal 与数据协议不变，只增加 coarse full-mask head、prediction-only 提示和 zero-init bounded residual。

这一方案相比旧的边界头设计更符合限制，也比 CPAR 的宽泛描述更具体：CPAR 可以视为研究方向，BCDH-R 是它的可实现、可消融版本。

## 主要原始资料

- [BASNet: Boundary-Aware Salient Object Detection, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Qin_BASNet_Boundary-Aware_Salient_Object_Detection_CVPR_2019_paper.html)
- [PointRend: Image Segmentation as Rendering, CVPR 2020](https://arxiv.org/abs/1912.08193)
- [UNet++: A Nested U-Net Architecture for Medical Image Segmentation, DLMIA 2018](https://arxiv.org/abs/1807.10165)
- [UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation, ICASSP 2020](https://arxiv.org/abs/2004.08790)
- [PraNet: Parallel Reverse Attention Network for Polyp Segmentation, MICCAI 2020](https://arxiv.org/abs/2006.11392)
- [Gated-SCNN: Gated Shape CNNs for Semantic Segmentation, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Takikawa_Gated-SCNN_Gated_Shape_CNNs_for_Semantic_Segmentation_ICCV_2019_paper.html)
- [SegFix: Model-Agnostic Boundary Refinement for Segmentation, ECCV 2020](https://arxiv.org/abs/2007.04269)
- [Boundary Loss for Highly Unbalanced Segmentation, MIDL 2019](https://proceedings.mlr.press/v102/kervadec19a.html)


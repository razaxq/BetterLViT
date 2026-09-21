# FSDR and RACE defence speaker notes

Updated 22 September 2026.

## Slide 1

English speaker script
Today I present FSDR and RACE for text-guided infection segmentation in chest X-rays. The two contributions are FSDR and complete RACE. The main evidence now includes four configurations and three matched training seeds, with all headline results measured on the Test split.

中文讲稿
本次答辩介绍 FSDR and RACE 胸片感染分割。两项贡献是 FSDR 和完整 RACE。主证据已更新为四组配置、三个匹配训练种子，主要成绩均来自测试集。

Sources (snapshot: 22 September 2026)
docs/PAPER_RESULTS.md

## Slide 2

English speaker script
FSDR separates semantic refinement from spatial detail. Complete RACE combines report-based routing with its auxiliary training objectives. The three-seed mean IoU improvements are 0.5058 percentage points for FSDR over PLAM and 0.2428 points for adding complete RACE to FSDR. The combined improvement is 0.7487 points over the matched PLAM control. These are module comparisons under one shared recipe.

中文讲稿
FSDR 将语义和细节分开精炼。完整 RACE 包括报告引导路由及配套辅助训练目标。三种子平均 IoU 增益分别为：FSDR 相对 PLAM +0.5058 个百分点，完整 RACE 加入 FSDR +0.2428 个百分点，组合相对匹配 PLAM +0.7487 个百分点。监督是完整 RACE 的组成部分，不作为独立创新点。

Sources (snapshot: 22 September 2026)
docs/results/stage1_overall_20260915/FINAL_REPORT.md
docs/results/stage1_overall_20260915/FINAL_RESULTS.json

## Slide 3

English speaker script
The original CNN and transformer paths run first. RACE then replaces each decoder-facing CNN skip C1 to C4 with its routed version. It does not feed routed features back into the encoder or transformer. FSDR replaces PLAM inside each of up4, up3, up2 and up1. It receives the routed skip, reconstructed ViT feature, upsampled decoder feature and text. It returns Y and D-prime for concatenation and two convolutions. The first decoder stage starts from bottleneck C5, and subsequent stages use the previous decoder output.

中文讲稿
CNN 与 Transformer 主路径先完成计算，RACE 随后精炼送往解码器的 C1–C4 跳跃特征，不将路由结果反馈进编码器或 Transformer。FSDR 位于 up4、up3、up2、up1 内部，替换 PLAM，接收路由跳跃、ViT 重建、上采样解码器与文本输入，输出 Y 与 D′ 后拼接卷积。首级由瓶颈 C5 开始，之后各级使用上一级解码器输出。

Sources (snapshot: 22 September 2026)
nets/LViT.py:471-525
nets/race_fuse.py
docs/FSDR.md

## Slide 4

English speaker script
This comparison concerns the PLAM decoder fusion mechanism and the FSDR extension. The matched PLAM baseline uses our common training recipe and frozen CXR-BERT. It is not an unchanged reproduction of the entire official LViT training pipeline.

中文讲稿
这里比较的是 PLAM 解码融合机制与 FSDR 扩展。匹配 PLAM 基线使用本项目统一训练配方和冻结 CXR-BERT，不等同于官方 LViT 整体流程原样复现。

Sources (snapshot: 22 September 2026)
nets/LViT.py
docs/FSDR.md
docs/results/plam_baseline_audit_20260916/AUDIT.md

## Slide 5

English speaker script
The upper path is the PLAM decoder operation. The lower path shows the combined model: RACE first produces the CNN skip C-prime, and FSDR receives C-prime, V, D and text separately. FSDR outputs Y and D-prime, which feed concatenation and two 3 by 3 convolutions. FSDR-only experiments use the original C instead of C-prime. Only the deeper up4 and up3 stages adapt D; at up2 and up1, D-prime equals D.

中文讲稿
上方为 PLAM 解码操作，下方为组合模型：RACE 先得到 C′，FSDR 分别接收 C′、V、D 和文本，输出 Y 与 D′ 后拼接并进行两层 3×3 卷积。仅 FSDR 对照直接使用原 C。只有 up4、up3 自适应精炼 D，up2、up1 的 D′=D。

Sources (snapshot: 22 September 2026)
nets/LViT.py:156-166
nets/LViT.py:503-525
nets/eppa.py

## Slide 6

English speaker script
C-star denotes the input CNN skip, routed when RACE is enabled. Haar decomposition separates low and high frequencies. Low-frequency skip, ViT and decoder signals plus text generate channel and region corrections and semantic support S. S modulates the high-frequency detail branch, which combines local and contextual convolutions. The final output adds bounded residuals to the original skip. Adaptive filtering, detailed in the backup slide, produces D-prime and R-adaptive at up4 and up3.

中文讲稿
C* 表示输入跳跃特征，启用 RACE 时为 C′。Haar 分解低高频，跳跃、ViT、解码器低频与文本形成通道/区域修正和语义支持 S，S 引导由局部与上下文卷积组成的高频细节分支。输出以原跳跃为基础叠加受限残差。备份页展示的自适应滤波在 up4、up3 生成 D′ 和 R_adaptive。

Sources (snapshot: 22 September 2026)
nets/eppa.py:679-777
docs/FSDR.md

## Slide 7

English speaker script
All four configurations use the same data split, input size, training budget and optimisation recipe. The three seeds are paired. Eight new runs and four eligible previous runs form the 12-model comparison. The learning rate decays once from 3e-4 to 1e-6. Adam weight decay is 1e-4. Dice and Focal each have weight 0.5, with Focal gamma 2. Complete RACE has auxiliary weight 0.05. Validation macro IoU selects the checkpoint, and Test uses strict probability greater than 0.5.

中文讲稿
四组模型采用同一数据划分、输入尺寸、训练预算与优化配方，种子一一匹配。八次新增训练加四个匹配既有模型组成十二模型整体消融。学习率单周期从 3e-4 降至 1e-6，Adam 权重衰减 1e-4，Dice/Focal 各 0.5，gamma=2。完整 RACE 辅助权重 0.05。验证集 macro IoU 选 Best，测试严格使用概率 >0.5。

Sources (snapshot: 22 September 2026)
docs/results/stage1_overall_20260915/FINAL_REPORT.md
docs/results/stage1_overall_20260915/FINAL_RESULTS.json

## Slide 8

English speaker script
Masked pooling of frozen text feeds the slot head. Six location probabilities combine with the aligned anatomical basis to form P. Each CNN skip generates visual evidence E and a local residual R(C). Zone pooling compares visual evidence with report slots to produce agreement A. P, E and A jointly gate a signed correction, with strength bounded by 0.15 and initial strength zero. Complete RACE includes its auxiliary training objectives, while inference uses predicted slots rather than parser targets.

中文讲稿
冻结文本经掩码均值池化后进入槽位头，六个位置概率与对齐解剖基底生成 P。CNN 跳跃生成视觉证据 E 和残差 R(C)，区域池化比较视觉与报告槽位得到 A。P、E、A 共同控制有符号修正，强度上限 0.15、初始为零。完整 RACE 包含辅助训练目标，但推理使用预测槽位，不使用解析器目标。

Sources (snapshot: 22 September 2026)
nets/race_fuse.py
race_semantics.py
docs/results/stage1_overall_20260915/FINAL_REPORT.md
docs/results/stage1_overall_20260915/FINAL_RESULTS.json

## Slide 9

English speaker script
The four configurations isolate FSDR and complete RACE at the whole-module level. The combined model averages 76.2262 percent IoU and 84.7112 percent Dice. Its gains over the matched PLAM baseline are 0.7487 and 0.6861 percentage points respectively. The displayed uncertainty is sample standard deviation across three training seeds, not a confidence interval. The historical 74.8076 percent PLAM result used a different recipe and is not the baseline for these increments.

中文讲稿
四组配置在完整模块层面分离 FSDR 与 RACE 的效果。组合模型平均 IoU 76.2262%、Dice 84.7112%，相对匹配 PLAM 分别提高 0.7487、0.6861 个百分点。± 表示三个训练种子的样本标准差，不是置信区间。历史 PLAM 74.8076% 使用不同配方，不作为本页增量基线。

Sources (snapshot: 22 September 2026)
docs/results/stage1_overall_20260915/FINAL_REPORT.md
docs/results/stage1_overall_20260915/FINAL_RESULTS.json

## Slide 10

English speaker script
Each row compares models trained with the same seed. FSDR improves PLAM both with and without complete RACE. Complete RACE improves both PLAM and FSDR. Every listed paired contrast is positive on all three seeds. This does not establish a universal effect or statistical significance. The three-seed study does not split RACE routing from auxiliary supervision, and the combined result does not prove super-additive synergy.

中文讲稿
每行都使用同种子配对。FSDR 在有无 RACE 时均改善 PLAM；完整 RACE 在 PLAM 与 FSDR 上均带来增益。列出的五种比较在三个种子上全部为正，但不据此声称普适性或统计显著。三种子研究没有拆分 RACE 路由与辅助监督，也不声称组合具有超加性协同。

Sources (snapshot: 22 September 2026)
docs/results/stage1_overall_20260915/FINAL_REPORT.md
docs/results/stage1_overall_20260915/FINAL_RESULTS.json

## Slide 11

English speaker script
All rows report the same 2,113 Test images with per-image macro metrics and a strict 0.5 threshold. GuideDecoder is the LanGuideMedSeg implementation, retrained for 100 epochs with best epoch 26. DD-CMD was retrained for 160 epochs with best epoch 115. Each uses seed 1219 and its own principal training recipe. MMI-UNet uses the author checkpoint, whose training filename list was not independently verified, so overlap cannot be excluded. Ours is a three-seed mean. DD-CMD is slightly higher, by 0.0732 IoU percentage points. These are contextual external comparisons, not matched module ablations or proof of state-of-the-art performance.

中文讲稿
各行在同一 2113 张测试图上使用逐图 macro 指标及严格 >0.5 阈值。GuideDecoder 即 LanGuideMedSeg 实现，训练 100 轮、Best 为 26；DD-CMD 训练 160 轮、Best 为 115。两者使用 seed1219 及各自主要配方。MMI 使用作者权重，其训练文件清单未独立核查，因此不能排除重叠。我们展示三种子均值。DD-CMD 的 IoU 略高 0.0732 个百分点。这是外部参照，不是模块因果消融，也不据此声称 SOTA。

Sources (snapshot: 22 September 2026)
docs/results/text_guided_baselines_20260920/FINAL_REPORT.md
docs/results/text_guided_baselines_20260920/FINAL_RESULTS.json
docs/results/external_methods_20260920/full_test/TEST_RESULTS.json

## Slide 12

English speaker script
The completed primary study supports FSDR and complete RACE as two contributions under the shared experimental protocol. The combined model gains 0.7487 IoU percentage points over matched PLAM and all paired contrasts are positive across the three seeds. External DD-CMD remains slightly higher. Independent-cohort evaluation and repeated controls separating the RACE route from its auxiliary training remain future work.

中文讲稿
已完成的主实验支持 FSDR 和完整 RACE 在统一协议下作为两项贡献。组合相对匹配 PLAM 提高 0.7487 个 IoU 百分点，三种子所有配对比较均为正。外部 DD-CMD 仍略高。后续需要独立队列验证，以及重复的路由/辅助监督机制对照。

Sources (snapshot: 22 September 2026)
docs/results/stage1_overall_20260915/FINAL_REPORT.md
docs/results/stage1_overall_20260915/FINAL_RESULTS.json

## Slide 13

English speaker script
Thank you. I welcome questions about the two modules, the matched ablations and the interpretation of the completed results.

中文讲稿
谢谢。欢迎就两个模块、匹配消融及最终结果的解释提问。

Sources · 15 September 2026


## Slide 14

English speaker script
The backup slides contain detailed fusion results, the FSDR formulation, the separate learning-rate study, fixed-model RACE diagnostics, evaluation protocol and report-target binding comparisons.

中文讲稿
备用页提供详细融合成绩、FSDR 公式、独立学习率实验、RACE 固定模型诊断、评估协议及报告目标绑定对照。

Sources · 15 September 2026


## Slide 15

English speaker script
This table retains all 12 primary results, including the weaker runs. The combined model achieves IoU values of 76.2485, 76.2562 and 76.1738 percent. Each Best epoch comes from validation selection. We report the mean over every seed rather than choosing a winning seed for the quantitative headline.

中文讲稿
本页保留全部十二个主实验模型，包括较弱种子。组合模型三个 IoU 为 76.2485%、76.2562%、76.1738%。Best 轮次均由验证集选择，量化主结论使用全部种子均值，不挑选最优种子作为成绩。

Sources (snapshot: 22 September 2026)
docs/results/stage1_overall_20260915/FINAL_REPORT.md
docs/results/stage1_overall_20260915/FINAL_RESULTS.json

## Slide 16

English speaker script
This branch operates at the deeper decoder stages up4 and up3. Low-frequency CNN, reconstructed ViT and decoder features predict separate spatial mixture weights for decoder smoothing and skip high-pass correction. Each mixture selects among identity, 3 by 3 blur and 5 by 5 blur. The decoder uses a bounded residual low-pass update. The skip receives a bounded high-pass residual, added to Y. Shallow stages retain the base Haar semantic-detail branches but bypass these adaptive corrections.

中文讲稿
此分支位于较深的 up4、up3。CNN、ViT 重建和解码器低频共同预测两组空间混合权重，分别控制解码器平滑和跳跃高通修正。滤波组由恒等、3×3 模糊、5×5 模糊组成。解码器使用受限低通残差更新，跳跃高通残差加入 Y。浅层仍保留基本 Haar 语义/细节分支，只跳过这项自适应修正。

Sources (snapshot: 22 September 2026)
nets/eppa.py:107-314
nets/eppa.py:679-777
docs/FSDR.md

## Slide 17

English speaker script
This earlier single-cycle study uses seed 1219 and the matched auxiliary-only control. It separates the routing increment from auxiliary training. The route alone has a positive point estimate but an image-paired confidence interval that crosses zero. The repeated three-seed study evaluates the complete module. We do not assign its entire gain to routing.

中文讲稿
该单周期实验使用 seed1219 及匹配的仅辅助监督对照，用来分离路由增量。路由点估计为正，但逐图配对置信区间跨零。三种子主实验评价完整模块，不能把全部增益归因于路由。

Sources (snapshot: 22 September 2026)
Source: previous defence slide 10, original evidence notes follow.
English speaker script
The completed auxiliary-only control helps separate the two changes. Adding auxiliary objectives produces a Test IoU point estimate of plus 0.1915 percentage points. Enabling routing with those objectives held fixed adds 0.2472 points. Both image-paired confidence intervals cross zero. The complete configuration improves by 0.4387 points, with an interval from 0.1405 to 0.7276 points. This supports improvement of the full configuration on this split and seed. It does not establish that the route alone has a reliable positive effect, and the additive arithmetic is not a proof of independent causal contributions.

中文讲稿
已经完成的仅辅助监督对照将两项变化分开：辅助目标的 Test IoU 点估计为 +0.1915 个百分点，固定辅助目标再开启路由为 +0.2472 个百分点。两者图像配对区间均跨零。完整配置的增益为 +0.4387 个百分点，区间为 [+0.1405,+0.7276]。因此支持该划分和种子上的完整配置改善，但路由独立贡献的证据还不充分，不能把全部提升归因于路由。

Evidence details
10,000 paired-image bootstrap resamples; RNG seed 20260914. These are not patient-cluster or across-training-seed intervals.

Sources · 15 September 2026
outputs/p8_binding_aux_20260915/FINAL_RESULTS.json
outputs/p8_research_20260914/test_20260915/TEST_RESULTS.json
outputs/p8_research_20260914/FINAL_PAIR_REPORT.md

## Slide 18

English speaker script
The combined model has the highest validation and Test means among the four matched configurations. Validation IoU selects checkpoints. The headline result is Test IoU. The standard deviations describe variation across three training seeds, and the data-split limitations still apply.

中文讲稿
组合在四组匹配配置中具有最高验证和测试均值。验证 IoU 负责选权重，最终成绩以测试 IoU 为准。标准差表示三个训练种子之间的变化，数据划分限制仍然适用。

Sources (snapshot: 22 September 2026)
docs/results/stage1_overall_20260915/FINAL_REPORT.md
docs/results/stage1_overall_20260915/FINAL_RESULTS.json

## Slide 19

English speaker script
The dataset has 5,716 training, 1,429 validation and 2,113 Test images. The primary study selects Best by validation macro IoU and evaluates that fixed checkpoint using probability strictly greater than 0.5. The patient audit finds 434 recoverable IDs shared by Train and Val. No recoverable Test ID overlaps Train or Val, but anonymous samples prevent proving full patient independence. The Test set has prior research access, so it is not an untouched prospective evaluation.

中文讲稿
数据划分为训练 5716、验证 1429、测试 2113。主实验由验证 macro IoU 选择 Best，再以严格 >0.5 的阈值评估该固定检查点。审计发现 Train/Val 共享 434 个可恢复患者 ID；可恢复 Test ID 未发现交叉，但匿名病例使完整患者独立性无法证明。Test 曾用于研究，不能宣称完全未接触的前瞻评估。

Sources (snapshot: 22 September 2026)
docs/results/plam_baseline_audit_20260916/AUDIT.md
docs/results/stage1_overall_20260915/FINAL_REPORT.md
docs/results/stage1_overall_20260915/FINAL_RESULTS.json

## Slide 20

English speaker script
The repository integrates FSDR and complete RACE source with the result snapshots. Each experiment retains its original full source commit, tag and checkpoint hash. Integration did not retrain or re-evaluate models. The primary study contains 12 single-cycle models. External comparisons have their own source and recipe metadata. Reproduction follows those per-run records.

中文讲稿
仓库整合 FSDR 和完整 RACE 源码及结果快照。每项实验保留原始完整 SHA、标签和权重哈希，整合没有重新训练或推理。主实验为十二个单周期模型，外部对照有各自来源和配方。复现遵循每项运行记录。

Sources (snapshot: 22 September 2026)
docs/PAPER_RESULTS.md
docs/results/IMPORT_PROVENANCE.json
docs/results/stage1_overall_20260915/FINAL_REPORT.md
docs/results/stage1_overall_20260915/FINAL_RESULTS.json
docs/results/text_guided_baselines_20260920/FINAL_RESULTS.json

## Slide 21

Scope update (22 September 2026)
Historical comparison: seed 1219 only. This binding-repair diagnostic does not replace the complete-module three-seed ablation.
历史比较：仅 seed1219。绑定修复诊断不替代完整模块三种子消融。

English speaker script
Both RACE variants completed 80 epochs and used validation-selected checkpoints at epoch 75. Corrected binding improves Test IoU relative to original binding by 0.2981 percentage points, while the validation point estimate is lower by 0.1107 points and its interval crosses zero. We retain this split difference without retuning. The main contribution is the RACE mechanism. Repairing report-target assignment is implementation correctness and must not be presented as an additional architectural innovation. Both variants are reported so the correction is not hidden.

中文讲稿
两种 RACE 均完成 80 轮并使用 Val 选定的第 75 轮检查点。修正绑定相对原绑定的 Test IoU 增益为 0.2981 个百分点，而 Val 点估计低 0.1107 个百分点且区间跨零，两项结果同时保留，不据此回调参数。报告目标绑定修复属于实现正确性，不作为额外结构创新。

Sources · 15 September 2026
outputs/p8_binding_aux_20260915/FINAL_RESULTS.json
outputs/p8_research_20260914/test_20260915/TEST_RESULTS.json
outputs/p8_research_20260914/FINAL_PAIR_REPORT.md
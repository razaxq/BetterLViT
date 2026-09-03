# BetterLViT 实验台账

更新时间：2026-09-03（Australia/Sydney）

本文件是后续实验配置、Git 溯源、验证结果和推进状态的唯一人工维护台账。停止继续维护 `改动计划.xlsx`；旧工作簿仅作为历史快照保留。

## 记录规则

- 论文主口径为逐样本 macro Dice/IoU；不在本台账维护 FMISeg 论文的 micro 口径。
- 正式实验必须使用独立完整 Git SHA 和实验标签，运行环境、检查点 `source_git_commit` 与结果 JSON 必须一致。
- 机制筛选 pilot 只使用 validation：`AUTO_EVALUATE=0`、`TEST_SPLIT_ALLOWED=0`。未通过阶段门不得扩展或访问 Test。
- 当前及后续架构实验不使用 LoRA。Focal 可以使用；禁止使用 boundary loss，正式配置必须保持 `boundary_loss=0.0`。
- C0/P5 是已完成的 Dice/Tversky 配对验证：`0.5 * Dice + 0.5 * Tversky`，Tversky 的 FP/FN 权重为 `0.7/0.3`。这不代表后续主线禁用 Focal。

## 已完成的正式 Test 实验

所有指标均为 Test macro，样本数均为 2,113。

| ID | 配置 | 完整 Git SHA | 最佳 epoch | Dice/IoU @ 0.5 | 验证集选阈值 | Dice/IoU @ 选定阈值 | 结论 |
|---|---|---|---:|---:|---:|---:|---|
| B0 | Frozen CXR-BERT + original PLAM + Dice/BCE | `77f43ba26487c73b56da1080f83706eececb8203` | 120 | 0.835266 / 0.748076 | 0.656 | 0.840582 / 0.756348 | 正式完成 |
| A0 | CXR-BERT LoRA + original PLAM + Dice/BCE | `7e8fc4f9cc0602e98ce12dadcbc6d55052c4c331` | 109 | 0.831570 / 0.743281 | 0.626 | 0.834767 / 0.748608 | 正式完成 |
| A1 | CXR-BERT LoRA + original PLAM + Dice/Focal | `3e46ab48e4eb7c077e789d6d9afc0158cda18a73` | 80 | 0.835233 / 0.747126 | 0.560 | 0.837367 / 0.751033 | 正式完成；历史 Focal 实验 |
| A2 | CXR-BERT LoRA + FAM-EPPA V4-B + Dice/BCE | `994094f989069a4d5b7263dc4ad45d333e9d0ede` | 80 | 0.844099 / 0.757901 | 0.550 | 0.844966 / 0.759453 | 非 Focal 已完成结果中的最佳者 |
| A3 | CXR-BERT LoRA + FMISeg-adapted fusion + Dice/BCE | `1c0b7dae263b27860a3fd672431d6968f7beec0c` | 80 | 0.836136 / 0.749682 | 0.540 | 0.836527 / 0.750792 | 原理适配，不是完整 FMISeg 复现 |
| A4 | CXR-BERT LoRA + FAM-EPPA V4-B + Dice/Focal | `a1d40d3a305a34abc0e96885fae68532007485b2` | 80 | 0.844932 / 0.759909 | 0.516 | 0.845191 / 0.760446 | 当前历史最佳 macro；历史 Focal 实验 |
| A5 | CXR-BERT LoRA + FMISeg-adapted fusion + Dice/Focal | `4ef4bf0616975e4ebbab37ad660d63ee99b61216` | 139 | 0.835314 / 0.746882 | 0.562 | 0.838524 / 0.752085 | 原理适配；历史 Focal 实验 |
| A6 | Frozen CXR-BERT + TCSR V1 + original PLAM + Dice/BCE | `01cd2f7501804f07e04ad4595de2fc9e2c511ede` | 139 | 0.834810 / 0.747394 | 0.556 | 0.836082 / 0.749756 | LoRA=False；V1 未产生正增益 |
| A7 | Frozen CXR-BERT + TCSR V1 + FAM-EPPA V4-B + Dice/BCE | `75b6ed284f8e137fa6424040f578b8bc7afd1c5c` | 80 | 0.843212 / 0.757207 | 0.508 | 0.843273 / 0.757351 | LoRA=False；未形成稳健增益 |
| A8 | Frozen CXR-BERT + TCSR V2 + FAM-EPPA V4-B + Dice/Focal | `3e00c9016023f23fce690586150d6d524b3d6ecf` | 80 | 0.843154 / 0.757541 | 0.498 | 0.843147 / 0.757523 | 相对 A9 无稳健增益；历史 Focal 实验 |
| A9 | Frozen CXR-BERT + FAM-EPPA V4-B + Dice/Focal | `494ec30ce1cd94fac566b480877ea3391f5d64ed` | 150 | 0.842435 / 0.755412 | 0.548 | 0.843717 / 0.757799 | A4 的无 LoRA 配对；历史 Focal 对照 |

## TCSR validation-only pilot 结论

| Pilot | 机制 | 完整 Git SHA | 最佳 Val macro Dice/IoU @ 0.5 | 关键诊断 | 阶段结论 |
|---|---|---|---:|---|---|
| P1 / V2.1 | 边界保持非对称两跳路由 | `e00138529f5ce5ac3996f58e54f764274789081e` | 0.8177 / 0.7180 | 相对 A9 best Dice -0.0063；最小病灶 -0.0132；`x4→x3` gate 0.7047，`x3→x2` 关闭 | 失败；无 80 epoch/Test |
| P2 / V2.2 | `x3→x2` 单跳边界路由 | `58be4f092b4a1c2f141e510ff40766fb58d4519b` | 0.8153 / 0.7151 | 较 A9 epoch40 -0.0011；相对 A9 best -0.0088；最小病灶 -0.0206；gate 0.9898 饱和 | 失败；无 80 epoch/Test |
| P3 / V2.3 | 有界校准 gate | `874adf6b51e9e3a30b5295b36a1992b6c9098abb` | 0.8154 / 0.7149 | 较 A9 epoch40 -0.0009；相对 A9 best -0.0086；最小病灶 -0.0167；gate 0.3768，边界支持仍过密 | 失败；无 80 epoch/Test |
| P4 / V2.4 | 稀疏边界有界单跳路由 | `e07924b22956099207203f792eb3c36eb3dd11db` | 0.813003 / 0.712870 | 较 A9 epoch40 -0.003327；相对 A9 best -0.011044，95% CI [-0.014624, -0.007416]；最小病灶 -0.027197；gate 0.27998、focus 0.16654、delta RMS 0.00973；train-val gap约0.0054 | gate/稀疏性通过，但数值与小病灶失败；无 80 epoch/Test |

P4 验证导出：`/root/autodl-tmp/BetterLViT-paper-p4-tcsrv24/runtime_logs/p4_validation_acceptance/`。P1–P4 均未访问 Test。

## 当前非 Focal 配对验证：C0 → P5

### 锁定配置与溯源

| ID | 配置 | 正式 Git SHA | 标签 | GPU 预检 | 当前状态 |
|---|---|---|---|---|---|
| C0 | Frozen CXR-BERT + 无 TCSR + FAM-EPPA V4-B + Dice/Tversky；LoRA=False | `281bc1b40c782f6f42b8caa332e1fe1045cb29e0` | `pilot-c0-tversky-frozen-b16-seed1219-20260902` | 两次 batch16 整模输出一致；SHA-256 `8475bc883e5150be77eeff1541fd2afb32fa3ba1961f75ee06589ae75d1fd3a6`；峰值 allocated/reserved 15.079/16.441 GB | 40 epoch 状态 0；best epoch 29；validation macro Dice/IoU/precision = 0.810587/0.708856/0.824030；1429 样本；未访问 Test |
| P5 | 与 C0 相同，但启用 TCSR V2.5：`x3→x2` 单跳稀疏路由 + 训练期局部监督；LoRA=False | `7d3bfce1caf0444656abfa9e42531f99a495c539` | `pilot-p5-tcsrv25-tversky-frozen-b16-seed1219-20260902` | 模块 SHA-256 `3dc9354fbf4fd95761ac3ec9dd5658ac5faefd76def7bb3e6ac82b6c5e7bc060`；整模 SHA-256 `a20a520fdde460a1d5aca1ad0a3efd7a089a84a06b15fe120a00539549d1d407`；峰值 15.791/17.178 GB | 40 epoch 状态 0；best epoch 30；validation macro Dice/IoU/precision = 0.808773/0.706784/0.818106；数值门失败；其局部项包含边界监督，不符合最终方法约束；未访问 Test |

P5 局部监督：GT 边界 Dice 对齐 + `0.5 *` 边界外残差泄漏，总权重 0.02，前 5 epoch 线性 warmup；推理不需要标签。模块预检确认 V2.5 与 V2.4 推理误差为 0，重复误差为 0，`x1/x3/x4` 严格 identity，文本/跨尺度效应与所有必要梯度非零。该实验保留为诊断证据，但因使用了边界监督，不得成为最终方法或后续配置模板。

### 最终验证状态

- C0 和 P5 均已按 40 epochs、batch 16、seed 1219、deterministic、`drop_last=True` 完成，状态均为 0。
- 两份 best-checkpoint validation-only JSON 均核对通过：`split=validation`、`test_split_accessed=false`、1429 样本、提交来源一致且包含 precision。
- P5 相对 C0：macro Dice `-0.001814`，95% CI `[-0.004823, 0.001178]`；macro IoU `-0.002072`；macro precision `-0.005924`，95% CI `[-0.009139, -0.002782]`。
- 最小病灶四分位：Dice `+0.000351`，但 precision `-0.004655`，因此未满足“小病灶 Dice 与 precision 均不下降”。
- P5 gate `0.2583`，位于目标范围；`x1/x3/x4` identity 通过；边界外泄漏由训练中期约 `0.381` 降至 epoch 40 的 `0.280`；epoch 40 train-validation Dice gap 约 `0.0081`，未见异常。
- 结论：P5 数值门失败，不扩展到 80 epoch，不访问 Test；当前无活跃训练。

### P5 阶段门

P5 只与相同损失、seed、batch 和训练长度的 C0 比较，必须同时满足：

1. validation macro Dice 相对 C0 至少 `+0.002`；
2. 最小病灶四分位 Dice 不下降；
3. 最小病灶四分位 precision 不下降；
4. `x1/x3/x4` 保持严格 identity；
5. route gate 位于 `[0.15, 0.35]`；
6. 边界外残差泄漏随训练改善；
7. 无异常 train-validation gap。

任一关键项失败即停止，不扩展到 80 epoch，也不访问 Test。即使通过，也先记录并等待多 seed 正式计划。

## 运行与结果路径

- C0 runtime：`/root/autodl-tmp/BetterLViT-paper-c0-tversky/runtime_logs/`
- P5 runtime：`/root/autodl-tmp/BetterLViT-paper-p5-tcsrv25/runtime_logs/`
- 非 Focal GPU 预检：`/root/autodl-tmp/BetterLViT-paper-p5-tcsrv25/runtime_logs/preflight_20260902_nonfocal/`
- C0 完成后 validation 导出：`runtime_logs/c0_validation_acceptance/c0_best_val_0p5.json`
- P5 完成后 validation 导出与 C0 配对比较：记录在 P5 的 `runtime_logs/p5_validation_acceptance/`

## 决策日志

- 2026-09-02：C0/P5 配对阶段采用 Dice/Tversky，并以 FP 0.7、FN 0.3 强化假阳性惩罚。
- 2026-09-02：P4 失败；停止沿现有 TCSR 路由继续参数修补。
- 2026-09-02：C0/P5 完成确定性 GPU 预检、独立提交与标签锁定；C0 启动。
- 2026-09-02：停止维护 `改动计划.xlsx`，后续只更新本 Markdown 台账。
- 2026-09-02：C0 以状态 0 完成；best epoch 29 的 validation macro Dice/IoU/precision 为 0.810587/0.708856/0.824030，未访问 Test。随后按门控链启动 P5。
- 2026-09-02：P5 以状态 0 完成，但相对 C0 的 macro Dice 为 -0.001814，最小病灶 precision 为 -0.004655；尽管 gate、identity、泄漏改善和 train-validation gap 通过，数值门仍失败。停止 P5，不进行 80 epoch 或 Test。
- 2026-09-02：用户澄清 Focal 可以使用，禁止的是 boundary loss。后续恢复 Dice/Focal 候选，所有正式配置保持 `boundary_loss=0.0`；P5 因包含边界监督仅作为历史诊断实验。
- 2026-09-03：P6 的频率/分支分层诊断确认：粗头训练显著干扰共享主干，最终 residual 虽能救回大部分损失，却主要退化为全局负向置信度收缩；高局部纹理样本的救援幅度最弱。因此停止 BCDH-R 参数修补，转向梯度隔离、局部稀疏且正负平衡的 CDRR。
- 2026-09-03：锁定并启动 C2→P7 的 40-epoch validation-only 配对。两项均为 Frozen CXR-BERT、FAM-EPPA V4-B、Dice/Focal、LoRA=False、`boundary_loss=0.0`，明确禁止 Test；C2 为严格控制，P7 仅增加 CDRR V1。

## 研究反思与主线收敛

### 已被证据支持的结论

- FAM-EPPA V4-B 是目前最可信的架构贡献。A2 相对 A0、A4 相对 A1 均获得约 0.8–1.0 个百分点的 Test macro Dice 增益；在冻结文本编码器的 A6/A7 配对中，加入 FAM-EPPA 的 A7 也明显优于 original PLAM 的 A6。
- LoRA 不是必要增益来源：A0 低于 B0，冻结文本编码器的 A9 仍接近历史最佳。因此后续保持 Frozen CXR-BERT、LoRA=False。
- FMISeg 原理适配未形成可靠增益：A3/A5 均低于对应的 FAM-EPPA 实验；不能把局部借鉴描述成完整 FMISeg 复现或第二项已验证创新。
- TCSR V1–V2.5 尚不能作为性能型核心创新。P1–P4 依次修复路由关闭、gate 饱和、支持区域过密等机制问题；P5 又加入直接局部监督并换用 Dice/Tversky。最终 gate、identity、泄漏下降和泛化间隙均正常，但相对严格 C0 对照仍降低 macro Dice 和 precision。
- P5 的结果说明，即使加入额外局部监督，文本条件跳连路由仍未形成数值增益。当前路由与已有 FAM-EPPA/PLAM 的作用存在冗余或干扰，且更容易扩大预测区域、降低 precision；继续做 TCSR V2.6 式参数修补的研究价值很低。由于最终方法禁止 boundary loss，P5 的局部监督也不能继续沿用。

### 当前论文论点边界

1. 第一项架构创新可以使用 FAM-EPPA：自适应频率感知的解码器融合，并由多组配对结果支撑。
2. TCSR 目前只能作为系统探索及负结果，不能作为第二项已成立的核心创新。
3. Focal 可以作为主线损失；禁止 boundary loss。论文主口径继续使用逐样本 macro Dice/IoU。

### 建议的第二项创新转向（尚未开发）

停止继续修改文本路由，改为验证不需要边界目标的 **Cross-Scale Prediction Agreement Refiner（暂名 CPAR）**：

- 保留 Frozen CXR-BERT、LoRA=False、FAM-EPPA V4-B，恢复 Dice/Focal，明确 `boundary_loss=0.0`，不再启用 TCSR。
- 在最终分割头之外增加一个来自较粗解码尺度的轻量辅助分割头；两个头都只使用原始 segmentation mask 和 Dice/Focal，不生成边界图、距离图或边界损失。
- 使用粗、细两个预测的差异图作为不确定区域提示，让粗尺度上下文抑制孤立假阳性、细尺度分支保留局部结构。
- 修正分支采用零初始化、幅度受限的 residual logits；初始化时必须严格等价于无 CPAR 基线。
- 可加入预测一致性约束，但不能包含任何 boundary target/loss。推理阶段只保留主头与轻量修正，不需要 GT。
- 先建立相同 Dice/Focal、boundary loss 0 的 40-epoch validation-only 对照，再做 CPAR 配对；仍以 macro Dice `+0.002`、整体及最小病灶 precision 不下降为阶段门。通过后再做多 seed，最后才允许正式 Test。

该方向的论文组合将是：**FAM-EPPA（频率感知融合） + CPAR（跨尺度预测一致性修正）**。两者分别解决特征融合和输出校准，不依赖被禁止的 boundary loss，也比 EPPA + TCSR 更正交。

### BCDH 深入研究后的收敛设计（尚未开发）

CPAR 已进一步具体化为 **BCDH-R V1（Boundary-Conscious Dual-Head Refiner）**。完整论证、公式、代码接入点、排除项与实验门见 [`BCDH_RESEARCH.md`](BCDH_RESEARCH.md)。

- 不使用“分割头 + 边界头”，而使用 `112×112` 粗尺度完整 mask 头和 `224×224` 细尺度完整 mask 头；两头都只接受原始 segmentation mask 的 Dice/Focal 监督。
- 从粗、细预测本身生成 uncertainty、fine-only disagreement 和 coarse-only disagreement；不生成 GT boundary/distance/direction target。
- 在最终 logits 上使用零初始化、幅度受限的稠密 residual correction；初始化严格等价无 BCDH 基线。
- 建议首轮为 C1（A9 式严格控制）对 P6（BCDH-R V1）的 40 epoch validation-only 配对，batch16、seed1219、`boundary_loss=0.0`，不访问 Test。
- 阶段门除 macro Dice `+0.002` 外，还要求整体及最小病灶 precision 不下降、boundary F1 提升、Brier score 不恶化，并排除 residual 饱和或全图修正。
- “双头”或普通 deep supervision 本身不是充分创新；论文论点必须落在 prediction-only 的有符号跨尺度误差提示与 exact-identity bounded residual correction 上。
- 在 boundary F1/Brier/ECE 证据出现前，C 不解释为 *Calibrated*，避免未经验证的概率校准主张。

## BCDH-R V1 实施与当前训练：C1 → P6

### 锁定实现

- 开发提交：`8a0fe9b41bc0a5d515a353d52693cb06afeb3096`。
- BCDH-R 使用 `up2` 粗尺度完整 mask 头、原有 `up1` 细尺度完整 mask 头、prediction-only uncertainty/disagreement cues，以及零初始化、`delta_max=1.0` 的 bounded logit residual。
- P6 目标：`L_total = L_Dice/Focal(final) + 0.2 * L_Dice/Focal(coarse)`；两项都只使用完整 segmentation mask。
- C1/P6 均为 Frozen CXR-BERT、LoRA=False、FAM-EPPA V4-B、Dice/Focal `0.5/0.5`、Focal gamma `2.0`、`boundary_loss_weight=0.0`。

### 独立提交、标签与预检

| ID | 完整 Git SHA | 标签 | 配置 | 预检 |
|---|---|---|---|---|
| C1 | `106aeab700ee98653b5ec27994a0aa15b60dac07` | `pilot-c1-bcdh-control-frozen-b16-seed1219-20260902` | BCDH 关闭的严格配对对照 | 两次 batch16 前/反向输出 SHA-256 均为 `d51177e6a9c6b162766de6c0e2eabfdeb4c62d10e815d601c3efdf30d9eb2c86`；loss 均为 0.2612988949；reserved 峰值 16.441 GiB |
| P6 | `7217660e6ef16e2a495bab4f20c73403468f55e1` | `pilot-p6-bcdh-r-v1-frozen-b16-seed1219-20260902` | C1 + BCDH-R V1 + 0.2 粗头辅助完整-mask监督 | 两次 batch16 前/反向输出 SHA-256 与 C1 完全一致；loss 均为 0.3125534058；identity error 0；reserved 峰值 16.852 GiB |

模块单测同时确认：零初始化 identity error 0、重复前向误差 0、粗头/输出投影/第二步 refiner trunk 梯度均非零，且未使用 boundary target。

### 当前运行状态

- 训练链于服务器时间 `2026-09-02T20:41:07+08:00` 启动，顺序为 C1 训练 → C1 best validation 导出 → P6 训练 → P6 best validation 导出 → 配对比较。
- 两项均为 40 epochs、batch 16、seed 1219、deterministic CUDA、`drop_last=True`。
- `AUTO_TEST_EVALUATE=0`、`TEST_SPLIT_ALLOWED=0`；本轮禁止访问 Test。
- 最终状态：C1 与 P6 均完成 40 epochs，训练、best validation 导出和配对比较均成功；GPU 已空闲。
- 运行元数据：`/root/autodl-tmp/BetterLViT-paper-p6-bcdh/runtime_logs/bcdh_pair_current.env`。
- 两份 validation JSON 均为 1429 样本、提交一致、`test_split_accessed=false`；本轮未访问 Test。

### 预注册阶段门

P6 相对 C1 必须同时满足：validation macro Dice 至少 `+0.002`、整体 precision 不下降、最小病灶四分位 Dice/precision 均不下降、tolerance-2 boundary F1 提升、Brier 不恶化，并且 residual 统计不出现饱和或全图无差别修正。未通过则停止，不扩展训练、不访问 Test。

### 最终结果与结论

| ID | Best epoch | Val macro Dice | IoU | Precision | Recall | Boundary F1 tol=2 | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1 | 40 | 0.816330 | 0.716653 | 0.785124 | 0.891991 | 0.726298 | 0.023294 |
| P6 | 40 | 0.815978 | 0.716806 | 0.784709 | 0.890255 | 0.725759 | 0.022994 |

P6 相对 C1：macro Dice `-0.000352`（95% CI `[-0.002388, 0.001636]`）、IoU `+0.000152`、precision `-0.000414`、recall `-0.001735`、boundary F1 `-0.000539`；Brier 改善 `-0.000300`（越低越好）。最小病灶四分位 Dice `-0.002482`、precision `-0.003837`、boundary F1 `-0.002898`。

数值门失败。最终 BCDH residual 的平均绝对幅度为 `0.523526`，`98.87%` 为负修正；高 uncertainty 区的修正幅度 `0.486574` 反而低于其余区域 `0.532763`，说明分支主要学成全局单向收缩，而不是预期的不确定区域局部校正。P6 不扩展至 80/150 epochs，也不访问 Test；BCDH-R V1 只能保留为负结果与后续机制诊断依据。

## BCDH 频率诊断与 CDRR V1：C2 → P7

### P6 分支与高频分层诊断

- 诊断工具提交：`5e7e708e5732cc455aa867e48a16962b0e007fce`；只读取 P6 best checkpoint 与 C1/P6 validation 结果，`training_performed=false`、`test_split_accessed=false`，样本数 1429。
- P6 base 相对 C1：macro Dice `-0.018400`、precision `-0.055832`、recall `+0.040846`、Brier `+0.002975`，说明粗头辅助梯度已明显改变共享特征并扩大预测区域。
- P6 coarse 相对 C1：macro Dice `-0.003289`、precision `-0.008741`；P6 final 相对 C1则为 Dice `-0.000352`、precision `-0.000414`、Brier `-0.000300`。final 相对 base 救回 Dice `+0.018047`、precision `+0.055418`，但没有越过控制组。
- final residual 平均 `delta_mean=-0.456584`，只有 `9.48%` 像素为正修正；top-20% uncertainty 的绝对修正反而比其余区域低 `0.016780`。该分支主要学成全图负向收缩，而非局部双向纠错。
- Haar 高频最高四分位上 final 相对 C1 的 Dice 为 `+0.002512`；但归一化局部细节最高四分位为 `-0.002740`，且 final 对 base 的救援仅 `+0.009033`，远低于最低局部细节四分位的 `+0.026507`。高频能量口径不一致，支持后续同时报告 Haar 与归一化局部细节，而不能只挑有利分层。
- 诊断结果：`/root/autodl-tmp/BetterLViT-paper-p6-bcdh/runtime_logs/bcdh_frequency_diagnostic/p6_frequency_heads.json`。

### CDRR V1 设计

CDRR（Cross-scale Detail Reliability Refiner）不使用边界标签或 boundary loss，针对上述失败机制做三项结构限制：

1. 粗头和 refiner 输入从 segmentation trunk `detach`，阻断辅助目标对主干的梯度干扰；主分割基线仍由原 Dice/Focal 独立优化。
2. 使用细/粗尺度局部细节一致性、预测 disagreement 与 base uncertainty 构造 reliability，只选择确定性的 top-15% 像素支持集；支持集外 residual 严格为 0。
3. 支持集内先做加权中心化，再施加 `0.5 * tanh` 有界 residual，使正负修正均可发生，避免再次退化为全局单向阈值偏移；末层零初始化保证初始预测与控制组逐位相同。

开发提交：`45a2a5a6fa89ca80e399eab377005399a7fb3833`。模块检查确认 identity/repeat error 均为 0、支持率 `0.149992`、支持集外最大 residual 为 0、主干梯度隔离通过、refiner/粗头/base 梯度非零，正/负活动像素占比约 `45.58% / 54.42%`，未使用 boundary target。

### 独立提交、标签与正式预检

| ID | 完整 Git SHA | 标签 | 配置 | 正式提交 GPU 预检 |
|---|---|---|---|---|
| C2 | `06479cd3302a8ca11022eac0a6b62bdad097eb65` | `pilot-c2-cdrr-control-frozen-b16-seed1219-20260903` | Frozen CXR-BERT + FAM-EPPA V4-B + Dice/Focal；CDRR 关闭 | 两次 batch16 前/反向输出 SHA-256 均为 `d51177e6a9c6b162766de6c0e2eabfdeb4c62d10e815d601c3efdf30d9eb2c86`；loss `0.2612988949`；allocated/reserved 峰值约 15.014/15.047 GiB |
| P7 | `fe4547a0c60fe948c9a574d9afc7d691370aeb42` | `pilot-p7-cdrr-v1-frozen-b16-seed1219-20260903` | C2 + CDRR V1 + 0.1 粗头完整-mask辅助监督 | 两次输出 SHA-256 与 C2 逐位相同；loss `0.2869261503`；identity error 0；allocated/reserved 峰值约 15.224/15.439 GiB |

两项均锁定 40 epochs、batch 16、seed 1219、deterministic、`drop_last=True`、LoRA=False、`boundary_loss_weight=0.0`。训练链设置 `AUTO_TEST_EVALUATE=0`、`TEST_SPLIT_ALLOWED=0`，只允许 best-checkpoint validation 导出与配对比较。

### 当前运行状态与阶段门

- C2→P7 链于服务器时间 `2026-09-03T12:27:26+08:00` 启动；运行元数据：`/root/autodl-tmp/BetterLViT-paper-p7-cdrr/runtime_logs/cdrr_pair_current.env`。
- 当前状态为 `c2_training`。C2 成功训练并导出 1429 个 validation 样本后才会启动 P7；任一提交、训练、导出或比较失败都会停止，且不自动重启。
- P7 相对 C2 的主阶段门：validation macro Dice 至少 `+0.002`；整体及最小病灶四分位 precision 不下降；tolerance-2 boundary F1 提升；Brier 不恶化；归一化局部细节最高四分位 Dice/precision 均不下降；支持率接近 15%，支持集外 residual 为 0，且修正不退化为单向全局偏移。
- 阶段门失败则停止，不扩展 80/150 epochs、不访问 Test；通过后也先做多 seed validation 复核，再决定是否进入正式 Test。

### 历史结果归档

C1/P6 会话、清单、运行日志及频率诊断已按完整 Git SHA 归档到共享盘：

- `/root/autodl-fs/betterlvit_5090_migration/paper_experiment_artifacts/106aeab700ee98653b5ec27994a0aa15b60dac07/`
- `/root/autodl-fs/betterlvit_5090_migration/paper_experiment_artifacts/7217660e6ef16e2a495bab4f20c73403468f55e1/`

两项会话均核对为 2862 个文件、源/目标字节数完全一致，四个 Best/Last 检查点 SHA-256 全部匹配。服务器本地的两个旧会话副本已删除以释放训练盘空间，完整归档可恢复。

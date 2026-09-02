# BetterLViT 实验台账

更新时间：2026-09-02（Australia/Sydney）

本文件是后续实验配置、Git 溯源、验证结果和推进状态的唯一人工维护台账。停止继续维护 `改动计划.xlsx`；旧工作簿仅作为历史快照保留。

## 记录规则

- 论文主口径为逐样本 macro Dice/IoU；不在本台账维护 FMISeg 论文的 micro 口径。
- 正式实验必须使用独立完整 Git SHA 和实验标签，运行环境、检查点 `source_git_commit` 与结果 JSON 必须一致。
- 机制筛选 pilot 只使用 validation：`AUTO_EVALUATE=0`、`TEST_SPLIT_ALLOWED=0`。未通过阶段门不得扩展或访问 Test。
- 当前及后续 TCSR 实验不使用 LoRA，也不再使用 Focal。
- 当前非 Focal 目标为 `0.5 * Dice + 0.5 * Tversky`，其中 Tversky 的 FP/FN 权重为 `0.7/0.3`。

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
| C0 | Frozen CXR-BERT + 无 TCSR + FAM-EPPA V4-B + Dice/Tversky；LoRA=False | `281bc1b40c782f6f42b8caa332e1fe1045cb29e0` | `pilot-c0-tversky-frozen-b16-seed1219-20260902` | 两次 batch16 整模输出一致；SHA-256 `8475bc883e5150be77eeff1541fd2afb32fa3ba1961f75ee06589ae75d1fd3a6`；峰值 allocated/reserved 15.079/16.441 GB | 40-epoch validation-only 运行中 |
| P5 | 与 C0 相同，但启用 TCSR V2.5：`x3→x2` 单跳稀疏路由 + 训练期局部监督；LoRA=False | `7d3bfce1caf0444656abfa9e42531f99a495c539` | `pilot-p5-tcsrv25-tversky-frozen-b16-seed1219-20260902` | 模块 SHA-256 `3dc9354fbf4fd95761ac3ec9dd5658ac5faefd76def7bb3e6ac82b6c5e7bc060`；整模 SHA-256 `a20a520fdde460a1d5aca1ad0a3efd7a089a84a06b15fe120a00539549d1d407`；峰值 15.791/17.178 GB | 已锁定，等待 C0 状态 0 后启动 |

P5 局部监督：GT 边界 Dice 对齐 + `0.5 *` 边界外残差泄漏，总权重 0.02，前 5 epoch 线性 warmup；推理不需要标签。模块预检确认 V2.5 与 V2.4 推理误差为 0，重复误差为 0，`x1/x3/x4` 严格 identity，文本/跨尺度效应与所有必要梯度非零。

### 当前运行状态

- 活跃实验：`c0_frozen_freq_tversky`
- 服务器目录：`/root/autodl-tmp/BetterLViT-paper-c0-tversky`
- 会话：`C0_Test_session_09.02_11h59`
- 协议：40 epochs，batch 16，seed 1219，deterministic，`drop_last=True`
- 隔离：`AUTO_EVALUATE=0`、`TEST_SPLIT_ALLOWED=0`
- 启动后首次检查：epoch 3 已完成，epoch 4 运行中；GPU 96%，显存约 17.4/24.6 GB；无异常。

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

- 2026-09-02：用户确认后续不使用 Focal；采用 Dice/Tversky，并以 FP 0.7、FN 0.3 强化假阳性惩罚。
- 2026-09-02：P4 失败；不再围绕 Focal 对照迭代。
- 2026-09-02：C0/P5 完成确定性 GPU 预检、独立提交与标签锁定；C0 启动。
- 2026-09-02：停止维护 `改动计划.xlsx`，后续只更新本 Markdown 台账。

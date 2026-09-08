# BetterLViT 实验台账

更新时间：2026-09-09（Australia/Sydney）

本文件是后续实验配置、Git 溯源、验证结果和推进状态的唯一人工维护台账。停止继续维护 `改动计划.xlsx`；旧工作簿仅作为历史快照保留。

## 记录规则

- 论文主口径为逐样本 macro Dice/IoU；不在本台账维护 FMISeg 论文的 micro 口径。
- 正式实验必须使用独立完整 Git SHA 和实验标签，运行环境、检查点 `source_git_commit` 与结果 JSON 必须一致。
- 机制筛选 pilot 只使用 validation：`AUTO_EVALUATE=0`、`TEST_SPLIT_ALLOWED=0`。未通过阶段门不得扩展或访问 Test。
- 当前及后续架构实验不使用 LoRA。Focal 可以使用；禁止使用 boundary loss，正式配置必须保持 `boundary_loss=0.0`。
- C0/P5 是已完成的 Dice/Tversky 配对验证：`0.5 * Dice + 0.5 * Tversky`，Tversky 的 FP/FN 权重为 `0.7/0.3`。这不代表后续主线禁用 Focal。

## 最新执行记录（2026-09-09）

**R2首次检查：** 02:37的第1次短连接确认R2训练健康，完成4/80轮、正在第5轮，SHA/manifest一致、跟踪文件干净、未访问Test。第2–4轮整轮均值215.757秒，预计今天07:09:33（悉尼）完成；同一heartbeat已改约07:22进行最后一次检查。预算1/2，之后不再中途连接；完成预测及30分钟约束将在末次按实际时间核实。当前无R2最终结果，HF认证旧问题未重试。[快照和预测](../repro_archive/20260908/recipe_execution/README.md)。

**R2已提交：** 悉尼时间02:21:27发起R2后台训练，PID125897，来源 `9eca26de5b301099805530edbf5a1a8718bea662`，tag `experiment-r2-recipe-80e-seed1219-20260908-v2`。保持原增强和C4架构，仅采用注册的80轮单次余弦LR；没有根据R1数值改配置。当前只有启动回执，尚未检查健康，独立预算0/2；同一heartbeat已改到02:37首次检查，再据完整轮时预测末检。R1的HF认证问题单独待重新登录处理，模型均保留。[R2启动回执](../repro_archive/20260908/recipe_execution/r2_launch.json)。

R1完成80轮，Best78，Val macro IoU **0.723869**、Dice **0.822398**。相对C4 IoU仅+0.001073（+0.1073个百分点），95%配对图像区间[−0.003060,+0.005253]；未通过≥+0.003及正区间筛选。最小病灶组IoU +0.008239，第二面积组−0.008565；Precision下降、Recall上升，不据单一改善分组宣称成功。两次检查已用完，末次距训练结束约12分钟，符合约定。本轮未访问Test，停止R1扩展，按预注册计划接续独立R2。

R1来源 `f71e782940dc2a52b08111771f6e3d5f5a8d36c5`、80轮LR、逐图结果、原始日志和Best/Last来源均核对。HF凭据失效，本地认证与远端写入均401；五个待传文件的Xet清单及完整模型保留，重新登录后续传，不宣称上传已完成。系统盘有约12.6GB可用，R2接续不会依赖删除这些模型。[完整R1结果](../repro_archive/20260908/recipe_execution/r1_results/README.md)。

## 最新研究记录（2026-09-08）

**R1首次检查：** 21:29的第1次短连接确认R1训练健康，已完成4/80轮、正在第5轮，SHA与运行manifest一致，跟踪文件干净，未访问Test。第2–4轮整轮均值211.492秒，预计2026-09-09 01:56:03（悉尼）完成训练；同一heartbeat已预约02:09进行第2次也是最后一次检查。当前检查预算1/2，之后不再中途连接；最终必须核对真实结束时间与30分钟约束。R2仍未启动，当前没有完整新Val/Test结果。[预测与快照](../repro_archive/20260908/recipe_execution/README.md)。

**R1/R2配方实验：** 用户已授权按[正式实验计划](TRAINING_RECIPE_PLAN_20260908.md)执行。R1删除原90度倍数旋转/镜像分支并保留随机数消耗，R2把原10轮LR重启改为首轮3e-4至第80轮1e-6的单次余弦；分别与C4独立对照，均从头80轮、seed1219、batch16、224、原Dice/Focal、无LoRA/新监督/外部视觉分支。R1已于21:13:38提交后台（PID114892），仅有启动回执，尚未检查健康；R2准备完成、未启动。正式来源分别为 `f71e782940dc2a52b08111771f6e3d5f5a8d36c5` / `9eca26de5b301099805530edbf5a1a8718bea662`，tag分别为 `experiment-r1-recipe-80e-seed1219-20260908-v2` / `experiment-r2-recipe-80e-seed1219-20260908-v2`。9项回归检查通过，最终来源各两次真实Train批次五步CUDA结果一致，R2还与C4五步完全一致。R1检查预算0/2，首检21:29，再按实际整轮耗时预约末检；R1完成后继续R2，不以R1数值通过为前提。当前无新Val/Test结果。完成后按IoU≥+0.003、配对区间及非退化门槛筛选，再做匹配种子和最终Test；不把配方优化当第二项结构创新。完整来源、预检、启动与定时记录见[执行归档](../repro_archive/20260908/recipe_execution/README.md)。

**P12后复盘：** 已完成[训练策略复盘与数据审计](../repro_archive/20260908/strategy_review/复盘与下一步.md)。全部1429张Val mask及抽查32张Train mask本来已是224×224；当前resize与最近邻完全一致，排除这一步新增掩码边界变化。共用增强包含90度倍数旋转/镜像而报告token不变；P12七次学习率重启均伴随暂时Val回落，但两点都尚未被证明是最终退化原因。下一步建议限定为C4的胸片方向增强、单次学习率衰减两个独立对照，保持架构和原Dice/Focal，再按配对种子和最终Test验证。该复盘阶段只进行了研究/审计；后续启动状态以上方R1/R2记录为准，尚无新增Test访问。

**执行更新：** P12的80轮训练及自动Val导出已完成，Best epoch70，Val macro IoU/Dice为 **0.716820/0.816282**；C4为0.722796/0.821214。IoU下降0.005977（−0.598个百分点），配对图像95%区间[−0.010325, −0.001710]，最小面积组IoU下降0.008938，四项门槛均失败。停止当前中层相加配置，不启动已准备的C9、不扩150轮、不访问Test。检查总计2次，末次距实际训练结束约11分36秒，定时已删除。源码`5d09913d46863073cba93159af4ed61f88fdd96e`，完整证据见[P12最终结果](../repro_archive/20260908/visual_prior_execution/p12_results/README.md)。前置CXformer探针IoU0.530981优于同构DINOv2的0.475959，但该优势没有转化为本配置的完整BetterLViT增益。

P11 已完成用户指定的150轮，最佳仍为80轮，Test macro IoU/Dice为0.755661/0.842335；结果与原80轮Best一致。完整运行与逐图证据见[最终归档](../repro_archive/20260908/dual_grain_150_results/README.md)。延长训练没有带来增益。

针对“深入研究第二点方向”，已完成[冻结视觉预训练特征研究报告](../repro_archive/20260908/frozen_visual_prior/研究报告.md)。优先候选为CXformer-S，同结构自然图像DINOv2-S作预训练来源对照，DINOv3-S保留后续候选。计划从C4接入单处中层残差投影，约41,280个可训练参数，沿用原Dice/Focal；这是普通融合可行性方案，尚非独立创新或性能结果。报告记录直方图均衡化/分辨率混淆、固定权重版本、解析预算及Train/Val→配对种子→Test流程。

研究报告完成时仅进行了公开元数据审计；后续诊断与训练推进以上方执行更新为准。当前P12已得到完整BetterLViT单种子Val退化结果，无正增益或新增Test证据。

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
- 2026-09-03：共享目录必须保持在 20 GB 以下。完成存储审计和裁剪后，`betterlvit_5090_migration` 从 24,266,917,767 bytes 降至 18,559,777,853 bytes；训练盘可用空间从 4.9 GB 提升至 7.3 GB。后续不得再把完整训练会话无条件复制到共享盘。
- 2026-09-03：C2/P7 均完成 40 epochs、best validation 导出和配对比较，状态为 `complete`，未访问 Test。P7 相对 C2 的 macro Dice 显著下降 `-0.003694`，小病灶 Dice 下降 `-0.008555`，两个高频口径也未同时满足 Dice/precision 不下降；CDRR V1 数值门失败，不扩展训练。

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

### 运行状态与阶段门

- C2→P7 链于服务器时间 `2026-09-03T12:27:26+08:00` 启动；运行元数据：`/root/autodl-tmp/BetterLViT-paper-p7-cdrr/runtime_logs/cdrr_pair_current.env`。
- 最终状态为 `complete`：C2/P7 均完成 40 epochs、best-checkpoint validation 导出与配对比较；GPU 已空闲，训练/导出/比较日志无错误。
- P7 相对 C2 的主阶段门：validation macro Dice 至少 `+0.002`；整体及最小病灶四分位 precision 不下降；tolerance-2 boundary F1 提升；Brier 不恶化；归一化局部细节最高四分位 Dice/precision 均不下降；支持率接近 15%，支持集外 residual 为 0，且修正不退化为单向全局偏移。
- 阶段门失败则停止，不扩展 80/150 epochs、不访问 Test；通过后也先做多 seed validation 复核，再决定是否进入正式 Test。

### C2/P7 最终结果

两份结果均核对为 `split=validation`、`test_split_accessed=false`、1429 样本、提交来源一致、LoRA=False、`boundary_loss_weight=0.0`。C2 best epoch 40；P7 best epoch 39。

| ID | Val macro Dice | IoU | Precision | Recall | Boundary F1 tol=2 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| C2 | 0.816330 | 0.716653 | 0.785124 | 0.891991 | 0.726298 | 0.023294 |
| P7 | 0.812636 | 0.711743 | 0.787725 | 0.881132 | 0.724445 | 0.023394 |

P7 相对 C2：macro Dice `-0.003694`，95% CI `[-0.006475, -0.000939]`；IoU `-0.004911`；precision `+0.002601`；recall `-0.010859`；boundary F1 `-0.001853`；Brier `+0.000100`（变差）。最小病灶四分位 Dice `-0.008555`、IoU `-0.010914`、precision `+0.011374`、recall `-0.035319`、boundary F1 `-0.009366`。

高频最高四分位结果并不稳健：

- Haar/Laplacian 能量口径：Dice `-0.001610`，precision `+0.009266`，未通过；
- 归一化局部细节口径：Dice `+0.000302`，precision `-0.000873`，未通过。

CDRR 机制约束本身工作正常：支持率 `0.149992`，支持集外 residual 最大值为 0，活动像素正/负修正约 `53.88% / 46.12%`，平均绝对 residual `0.021526`，未再次退化为全图单向收缩。但它通过提高 precision 换取了明显 recall 损失，尤其小病灶 recall 下降 `-0.035319`，导致整体 Dice、边界 F1 与 Brier 均未改善。

结论：CDRR V1 数值门失败，不扩展至 80/150 epochs，不做多 seed 或 Test。该结果说明“梯度隔离 + 稀疏双向修正”解决了 BCDH 的机制退化，却没有解决支持区域选择对真阳性细节的误抑制；后续如继续研究，必须先做离线错误归因或重新设计 reliability，而不是直接调节 residual 幅度或训练轮数。

### 历史结果归档

C1/P6 会话、清单、运行日志及频率诊断已按完整 Git SHA 归档到共享盘：

- `/root/autodl-fs/betterlvit_5090_migration/paper_experiment_artifacts/106aeab700ee98653b5ec27994a0aa15b60dac07/`
- `/root/autodl-fs/betterlvit_5090_migration/paper_experiment_artifacts/7217660e6ef16e2a495bab4f20c73403468f55e1/`

两项会话均核对为 2862 个文件、源/目标字节数完全一致，四个 Best/Last 检查点 SHA-256 全部匹配。服务器本地的两个旧会话副本已删除以释放训练盘空间，完整归档可恢复。

### 服务器存储策略（20 GB 共享目录上限）

- 2026-09-03 审计前：共享目录 24,266,917,767 bytes，本地训练盘仅余 4.9 GB。
- 已从 6 个 8 月旧会话中删除可由 sibling Best 替代的 Last checkpoint；8 个历史 Best 全部保留，另保留 3 个 Last。删除前的 SHA-256、字节数与完整路径记录在 `/root/autodl-fs/betterlvit_5090_migration/runtime_logs/storage_optimization_20260903_deleted_legacy_last.sha256`。
- 已删除共享盘中未被训练使用、可重建的旧 `huggingface-cache`；当前训练只使用 `root_cache/huggingface`。
- 已删除本地无活跃进程使用的 `hf-upload-staging`、`wheelhouse-cu128` 和旧 `huggingface` 缓存。上传暂存区不是唯一结果来源；wheelhouse/缓存可重建。
- 审计后：共享目录 18,559,777,853 bytes（约 18.56 GB），低于 20 GB；本地训练盘可用 7.3 GB，P7 训练保持正常。
- 后续保留顺序：结果 JSON、manifest、Git SHA/标签、Best checkpoint 优先；Last 只在需要断点续训时保留。完整会话优先发布到已校验的 Hugging Face Bucket 或外部归档，不再默认写入共享盘。每次归档前必须先检查 20 GB 上限。

## RACE-Fuse V1：C3 → P8（80 epochs）

### 研究动机与机制

P6/P7 证明单纯在输出端做局部修补容易以 recall 换 precision，尤其伤害小病灶。RACE-Fuse（Report-Anatomy Consistency and Evidence Fusion）改为在四层 encoder skip 上融合报告位置语义与视觉病灶证据：冻结 CXR-BERT 的 token 特征学习六区位置槽和三类病灶数量槽；每个尺度独立预测视觉 evidence；只有“报告正证据 × 局部视觉证据 × 图文一致性”共同成立时才开启有界加性路由。未提及区域按 unknown 处理，不作为病灶负标签；路由强度零初始化，未学习前与控制网络逐位相同。

训练数据的随机旋转/翻转会同步作用于六区解剖基底，避免图像增强后文本方位与空间先验错位。RACE 不使用 LoRA，主目标固定 Dice/Focal `0.5/0.5`，辅助权重 `0.05`，`boundary_loss_weight=0.0`。

### 独立提交、标签与预检

| ID | 完整 Git SHA | 标签 | 配置 | 正式提交 GPU 预检 |
|---|---|---|---|---|
| C3 | `424f2824b737e4d311cdcb270bbc83e973dc067f` | `pilot-c3-race-control-frozen-b16-seed1219-80e-20260903` | Frozen CXR-BERT + FAM-EPPA V4-B + Dice/Focal；RACE 关闭 | 两次 batch16 输出 SHA-256 均为 `d51177e6a9c6b162766de6c0e2eabfdeb4c62d10e815d601c3efdf30d9eb2c86`；loss `0.2612988949`；allocated/reserved 峰值约 15.10/16.46 GiB |
| P8 | `48df2f185a28919e38e073322809d043e176d382` | `pilot-p8-race-fuse-v1-frozen-b16-seed1219-80e-20260903-restart2` | C3 + RACE-Fuse V1；aux weight 0.05；四尺度最大 residual strength 0.15 | 最终提交两次 batch16 输出 SHA-256 均与 C3 逐位相同；loss `0.2993894517`（含辅助项）；identity strength 0；allocated/reserved 峰值约 17.33/18.70 GiB；真实训练样本增强/六区基底数据契约检查通过 |

早期 P8 提交 `d3ac4ad377447ed2fdb9d3176ae9369c106a77d4` 在首次 GPU 预检发现辅助 mask 多余通道维，未启动训练、无实验结果；`edb719c6245e2005853b220e233e9fcc76fae405` 修复后通过两次合成预检，但最终正式提交另加入真实 DataLoader 契约检查，因此两者均不得作为 P8 结果提交。

### 当前运行状态与阶段门

- C3→P8 验证专用链于服务器时间 `2026-09-03T18:30:26+08:00` 启动；元数据：`/root/autodl-tmp/BetterLViT-race-p8/runtime_logs/race_pair_current.env`。
- 两项均锁定 80 epochs、batch 16、seed 1219、deterministic、`drop_last=True`、LoRA=False、boundary loss=0；`AUTO_TEST_EVALUATE=0`、`TEST_SPLIT_ALLOWED=0`。
- 最终链状态为 `complete`：C3/P8 均完成 80 epochs，两个 best-checkpoint validation 导出和配对 bootstrap 比较均成功；GPU 已空闲，日志无训练、CUDA、数据或检查点错误。
- 两组 Best/Last 均可严格加载，`source_git_commit` 分别匹配上表提交；均为 LoRA=False、boundary loss=0，且 Test 从未访问。
- P8 相对 C3 的预注册门：macro Dice 至少 `+0.002`；整体 precision 不下降；最小病灶四分位 Dice 与 recall 不下降；Brier 不恶化；RACE 路由强度/门控/视觉 evidence/一致性统计不得坍缩或退化为全图均匀增强。模型选择完成前不访问 Test。

### C3/P8 最终结果

两份结果均为 `split=validation`、`test_split_accessed=false`、1429 样本、阈值 0.5，且 best epoch 都是 80。

| ID | Val macro Dice | IoU | Precision | Recall | Boundary F1 tol=2 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| C3 | 0.821214 | 0.722796 | 0.814701 | 0.864661 | 0.739182 | 0.021477 |
| P8 | 0.823061 | 0.725566 | 0.813476 | 0.869796 | 0.743003 | 0.021082 |

P8 相对 C3：macro Dice `+0.001848`，95% CI `[-0.001263, 0.005013]`；IoU `+0.002770`；precision `-0.001225`；recall `+0.005135`，95% CI `[0.000921, 0.009447]`；boundary F1 `+0.003821`；Brier 改善 `-0.000394`，95% CI `[-0.000628, -0.000159]`（越低越好）。

最小病灶四分位（358 样本）Dice `+0.001947`、precision `+0.001469`、recall `+0.007525`，但区间均跨 0；boundary F1 `-0.002337`。高频最高四分位在两种预注册口径均为正：Laplacian Dice/precision `+0.005027/+0.002907`，归一化局部细节 `+0.002645/+0.000626`。

RACE 末轮四路 strength 为 `[-0.05793, 0.06121, 0.06121, 0.08668]`，gate mean `0.3169–0.3592`，visual evidence mean `0.7187–0.7852`，agreement mean `0.8095–0.8581`；机制没有坍缩，也未退化为全图同号增强。

结论：P8 获得方向一致的 Dice/IoU、recall、Brier 和高频分层改善，第一次避免了 P6/P7 的小病灶 recall 损失；但 macro Dice 距预注册 `+0.002` 门槛少 `0.000152`，整体 precision 又下降 `0.001225`，且 Dice CI 跨 0，因此严格数值门为失败。当前不得访问 Test 或直接进入 150 epochs；应先基于 validation 做机制归因，并用独立预注册改动验证能否保留 recall/Brier 增益同时恢复 precision。


## RACE-PE V1：C4 → P9（2026-09-06，80 epochs）

用户选择 RACE-PE 后新增独立实现：区域 presence head 与像素 extent head 分开；presence 对齐文本存在性，extent 监督完整 mask，其区域均值对齐真实占比。文本位置缺失或含糊时使用 unknown；没有加入 boundary loss、LoRA 或 Lovasz。P8 的失败结论不变。

| ID | 完整 Git SHA | Git tag | 配置 |
|---|---|---|---|
| C4 | `add4908a0d6f702b0a10c4581725b535543829b8` | `pilot-c4-race-pe-control-80e-seed1219-20260906` | Frozen CXR-BERT + FAM-EPPA V4-B + Dice/Focal |
| P9 | `8129c1f039ed77f78e70305aca0bb9708b3b56b1` | `pilot-p9-race-pe-80e-seed1219-20260906` | C4 + RACE-PE；aux=0.05；bounded strength=0.15 |

- 开发提交：`8b31af87c6d1d7666d21100fb8f0183af790ecce`；设计说明：`D:/BetterLViT/race_pe_work/docs/RACE_PE_V1_DESIGN.md`。
- 两项统一：80 epochs、seed1219、physical batch16、deterministic、drop_last=True、阈值0.5。新配对按 validation macro IoU 选检查点，前5 epochs不参与选择；历史实验仍保留其原选择规则。
- 冻结源码的真实训练 batch16 前/反向分别重复两次，均通过且各自损失相同。四次初始输出 SHA-256 均为 `246feaa997468b4696ac8d02772f479d38f9315814ba220fed614be7abaa39b7`；C4/P9 allocated 峰值分别约15.10/17.33 GiB。单像素病灶、空mask、未知/否定文本、无效区域、辅助监督消融和IoU检查点选择行为检查均通过。
- 训练集文本/坐标审计：5716样本，4030个有保守可解析方位；1111个可解析明确单侧病例中1070个与当前图像坐标侧别一致。保留当前约定，不能把矩形六区描述为真实肺分割。
- 新服务器：`connect.westb.seetacloud.com:21465`，RTX4090D，PyTorch2.9.1+cu128。系统盘约22GiB可用，输出放在系统盘；不删除历史产物。
- 已发起后台 C4→P9 链，PID2757。状态路径：`/root/race_pe_runs/c4_p9_20260906/chain.status`；源码：`/root/BetterLViT-race-pe-c4`、`/root/BetterLViT-race-pe-p9`。已核对状态 `c4_training`，C4第1/80轮至少完成100/357个batch，GPU利用率99%，暂无训练错误；P9尚未开始，尚无新验证结果。
- `TEST_SPLIT_ALLOWED=0`、`AUTO_TEST_EVALUATE=0`。完成训练后只导出validation并生成`c4_vs_p9.json`，不自动扩展训练。
- 初筛门：macro IoU至少+0.003；整体Dice/precision、最小病灶四分位Dice/recall不下降，Brier不恶化。报告IoU配对bootstrap CI；即使单种子过门也不代表稳定增益。
- 已注册但未启动：C5（原RACE、同IoU选择规则）、C6（匹配权重的像素辅助监督、无路由）、C7（全部PE辅助监督、无路由）。后续先完成机制归因，再预注册至少3个配对种子；方法锁定前不访问Test。


### C4/P9 最终结果（2026-09-07核查）

链于悉尼时间2026-09-07 01:02:07完成，状态`complete`。C4/P9均完成80 epochs，两个validation导出及配对比较成功；日志未发现Traceback/RuntimeError/CUDA OOM，GPU空闲。两份JSON均为1429样本、阈值0.5、seed1219、IoU选检查点、test_split_accessed=false，提交来源与预注册一致。未启动额外训练或访问Test。

| 实验 | best epoch | Val macro Dice | IoU | Precision | Recall | Brier |
|---|---:|---:|---:|---:|---:|---:|
| C4 | 80 | 0.821214 | 0.722796 | 0.814701 | 0.864661 | 0.021477 |
| P9 | 69 | 0.822059 | 0.724335 | 0.796023 | 0.889474 | 0.021950 |

P9-C4：IoU +0.001538（95% CI [-0.001819, 0.004953]），Dice +0.000846；precision -0.018679（95% CI [-0.021869, -0.015604]），recall +0.024813；Brier +0.000473（恶化）。最小病灶四分位Dice -0.001677，IoU -0.000281，precision -0.035094，recall +0.047812。

结论：`passes_single_seed_screen=false`，IoU未达到+0.003且CI跨0；整体precision、最小病灶Dice和Brier亦未过门。不能称为稳定增益，不自动扩展至150 epochs或Test。

诊断线索：P9 best-checkpoint验证导出的最后一个batch中，文本槽概率均值0.998699；训练末轮均值0.9991。存在文本分支趋向全阳性的强烈信号；这些是末batch/末轮快照，不代表已经完成全验证集机制归因。V1使用positive-only文本监督，可能允许全阳性退化，需要单独诊断而非直接重启训练。

本地原始结果：`D:/BetterLViT/outputs/race_pe_results_20260907/`。归档SHA-256 `15421e4266f010d58115a9411f10ed849757d82e62d966c4097aa2b71a11f91f`已与服务器一致核验。


### C4/P9 用户授权 Test 检查（2026-09-07）

用户获知validation初筛失败后明确要求“用测试集看看结果”，本次授权覆盖C4/P9的Test评估。使用原validation选定的C4 epoch80与P9 epoch69，阈值固定0.5，未重训、未根据Test选检查点或阈值。历史validation JSON中的`test_split_accessed=false`描述当时状态；截至本次已访问Test。

两组完整评估2113张Test图像，导出与比较成功；来源提交仍为C4 `add4908a0d6f702b0a10c4581725b535543829b8`、P9 `8129c1f039ed77f78e70305aca0bb9708b3b56b1`。冻结训练工作树未改动；独立评估脚本沿用validation导出器的模型加载、预处理与指标实现，记录脚本SHA-256、授权和Test访问标记。

| 实验 | Test macro Dice | IoU | Precision | Recall | Brier |
|---|---:|---:|---:|---:|---:|
| C4 | 0.842989 | 0.756396 | 0.839363 | 0.874378 | 0.016938 |
| P9 | 0.841777 | 0.755351 | 0.820142 | 0.895348 | 0.017546 |

P9-C4：Dice -0.001212（95%配对图像bootstrap CI [-0.003527, 0.001090]）；IoU -0.001045（CI [-0.003706, 0.001705]）；precision -0.019221（CI [-0.021928, -0.016600]）；recall +0.020970（CI [0.017751, 0.024230]）；Brier +0.000608（恶化）。Test未显示IoU增益，IoU区间跨0不能声称显著下降；precision下降明确，仍不支持稳定增益。原validation失败结论不变，未启动后续训练。

原始Test结果、评估脚本、授权与日志：`D:/BetterLViT/outputs/race_pe_test_20260907/`。归档SHA-256 `85dec863d637fa0fb41e8c44546109e501caff10583f9cbbfb6075068642338e`已与服务器匹配；脚本哈希与2113样本数核验通过。


### RACE-PE退化诊断（2026-09-07，训练/验证专用）

已完成P9 epoch69的完整1429张验证图像前向干预、5716张训练图像标签/区域审计及16张验证图像的梯度探针。本次未重训或访问Test；此前用户授权的Test访问记录不变。原模型验证IoU精确复现0.7243345893510442。

- 确认文本全阳性退化：训练位置标签7894正例、0负例、26402未知。masked positive-only BCE允许全1解。完整验证8574个文本槽全部>0.99，均值0.998706。epoch1/20/69末验证batch槽均值约0.906/0.994/0.999。
- 仅将RACE槽设为全1或批内打乱，分别仅改变全部71701504验证像素中的20/16个二值预测，IoU均约0.724333。主LViT文本路径未改变，不能推断整个网络不使用文本。
- 关闭推理路由：IoU0.724149、precision0.802177、recall0.881141；相对native的precision回升0.006154，但仍比C4低0.012524。推理消融不等于独立训练的辅助监督消融，尚不能分离训练期间各分支的因果影响。
- 训练集无增强几何审计中，7894个报告派生正区域有1522个mask为空（19.28%）。固定图像矩形分区与肺部报告语义存在冲突；presence mask BCE要求0，文本positive consistency要求1。不能据此断言报告或mask错误。
- 验证四尺度视觉presence在真实空区域的误报率为99.02%/98.15%/88.06%/77.61%。Extent有一定前景/背景区分，但presence未有效校准。
- 4×4张验证图像的首层梯度探针中，辅助/主梯度范数比0.063–0.176，2/4组余弦为负；这是有限线索，不足以证明辅助梯度普遍主导训练。

后续优先修复文本目标可辨识性，将报告“提及”与疾病“存在”区分；移除或屏蔽与mask冲突的固定矩形正一致性。在小规模可学习性验证后再做匹配消融，不直接延长训练。未启动任何新训练。

完整报告与原始证据：`D:/BetterLViT/outputs/race_pe_diagnosis_20260907/诊断报告.md`。诊断归档SHA-256 `7e1544611e33c59142af43e9e1d92cdde983d156ce00f3dad4aa9849eb777dea`已与远端一致校验。


### RACE-PE V2最小修复与可学习性检查（2026-09-07）

V1的区域文本BCE只有正例、没有负例，允许全阳性解；报告区域与图像六个矩形区域也存在不一致。V2将监督定义为“编码器实际看见的文本明确提及哪些区域”，未提及作为这个语言任务的负例，不解释成疾病不存在；移除报告到视觉presence的正一致性损失。路由改为提及概率×视觉presence×像素extent。历史V1实现及C4/P9结果保持独立。

## 验证结果

- 历史V1行为检查和V2正/负提及、冲突空区域的梯度方向检查通过。
- 两次batch16完整CUDA前向/反向完全一致：loss 0.5188546180725098；输出SHA256 `246feaa997468b4696ac8d02772f479d38f9315814ba220fed614be7abaa39b7`；峰值分配17.33GiB，保留18.70GiB。
- 冻结CXR-BERT，仅训练随机初始化文本头，seed1219，固定40epoch，不按Val挑选epoch。
- Train共240种编码器可见模板，留出48种；拟合使用5200张图像的文本，Val1429张。

| 文本区域提及任务 | macro F1 | macro specificity | BCE |
|---|---:|---:|---:|
| Val正常文本 | 0.999180 | 0.996369 | 0.005928 |
| Train留出模板 | 0.984263 | 0.962965 | 0.101503 |
| Val循环打乱文本 | 0.615489 | 0.470361 | 3.322377 |
| Val全阳性预测 | 0.716333 | 0 | 4.233776 |

这些F1是规则派生的文本提及标签得分，并非分割Dice，也不是人工标注的临床语义准确率。常见报告模板在Train与Val间可以重复，因此额外按可见模板留出；留出分数按原始报告唯一项计算，同一可见模板可能有多个原始报告。该实验没有联合分割梯度，不能排除联合训练再次退化。

## 来源与边界

代码 `2f66ba898517dbee876ff6da735ac342943a08a2`，分支 `paper/race-pe-v2-development`，标签 `diagnostic-race-pe-v2-mentions-20260907`。本地与远端代码一致，远端仅有未跟踪的datasets软链接，指向 `/root/autodl-tmp/datasets`。运行环境见runtime.json。

原始归档SHA256 `3c55b851d1c7dc1c7c53311c78bbbdb2086d3fc77060344af7ffe0981445ec55`，已与远端核对。诊断文本头仅供复查，不自动用于后续分割初始化。

本次未访问Test，历史C4/P9已授权的Test访问事实不变。未启动新的完整分割训练，所有检查已结束，GPU空闲。

下一步应在Train/Val做匹配训练消融，分离“辅助监督”和“实际路由”的作用，再按预注册IoU/precision门槛决定是否扩大实验。固定矩形与肺解剖的错位仍是风险；文本头可学不等于路由可带来稳定增益。


### C8→P10 V2验证消融已启动（2026-09-07）

用户明确要求“启动”。悉尼时间2026-09-07 04:20:47启动后台链，先C8辅助监督，再P10辅助监督+路由，各80epoch、seed1219、batch16、Val IoU选择、阈值0.5、确定性训练。从头训练，不载入诊断文本头。复用已完成C4验证导出作无辅助监督基线。当前只是启动记录，不代表已完成或有增益。

- C8：`21606e02c2d64ae950c0f243f55163f58cbedf83`，tag `pilot-c8-race-pe-v2-80e-seed1219-20260907`。
- P10：`2f33a71219ed2beb339db45d702ecc36a931e0c8`，tag `pilot-p10-race-pe-v2-80e-seed1219-20260907`。
- 两组独立冻结manifest/工作树，仅路由开启与否不同；启动前batch16检查通过，初始loss0.5188546180725098和输出SHA完全一致。C8/P10峰值分配15.51/17.33GiB。
- 已实际观察C8第1轮60/357batch、GPU86%、显存17684MiB，无异常；P10等待自动接续。链PID29701，训练PID29710。
- 每组成功训练后自动导出完整Val，最终比较C4→C8、C4→P10、C8→P10。主门槛为P10对C4 macro IoU≥+0.003及既定非退化条件；路由归因还需C8→P10的正IoU区间且precision不退化。单种子不能证明稳定增益。
- 本轮`AUTO_TEST_EVALUATE=0`、`TEST_SPLIT_ALLOWED=0`，不会自动扩展或访问Test。此前C4/P9已访问Test记录保留。
- 远端输出：`/root/race_pe_v2_runs/c8_p10_20260907`。系统盘启动前可用20056121344字节，预计两组Best/Last加临时写入空间足够；不增加共享fs占用。

本地启动快照：`D:/BetterLViT/outputs/race_pe_v2_launch_20260907/launch_snapshot.json`。正式完成需核对chain.status、两个Val导出、三个比较JSON及来源提交。预注册说明见V2工作树`docs/RACE_PE_V2_SEGMENTATION_PILOT.md`。


### C8/P10 V2验证消融完成（2026-09-07）

悉尼时间14:51:06训练链complete，两组80轮及全部1429张Val导出、三组比较均完成。两组最佳检查点均为epoch80。远端冻结工作树跟踪文件干净；JSON来源与C8 `21606e02c2d64ae950c0f243f55163f58cbedf83`、P10 `2f33a71219ed2beb339db45d702ecc36a931e0c8`一致。GPU已空闲，未启动新实验，本轮未访问Test；此前C4/P9 Test访问记录不变。

| 实验 | Val macro IoU | Dice | Precision | Recall | Brier |
|---|---:|---:|---:|---:|---:|
| C4基线 | 0.722796 | 0.821214 | 0.814701 | 0.864661 | 0.021477 |
| C8仅辅助监督 | 0.726347 | 0.823727 | 0.818790 | 0.864762 | 0.021008 |
| P10辅助监督+路由 | 0.725594 | 0.823281 | 0.808676 | 0.876272 | 0.021305 |

C8-C4：IoU+0.003551，配对图像bootstrap95%CI[0.001131,0.006045]；precision+0.004088；最小病灶四分位Dice+0.002404、recall+0.002051；Brier降低。通过预注册单种子筛选。但这是同一seed的图像抽样区间，不证明跨训练种子的稳定增益。

P10-C4：IoU+0.002798，CI[-0.000625,0.006320]，未达+0.003门槛；precision-0.006025；最小病灶Dice-0.000826，筛选失败。P10-C8：IoU-0.000753，CI[-0.003779,0.002389]，不能声称显著IoU下降；precision-0.010113，CI[-0.013085,-0.007154]；recall+0.011510；Brier+0.000297。因此未证明路由带来增益，仍有扩大预测范围的表现。优先进一步核验C8辅助监督路径，而非延长P10或宣称完整RACE-PE创新成功。

结果目录 `D:/BetterLViT/outputs/race_pe_v2_results_20260907/`；下载归档SHA256 `4baa2e639eb43e4d966542fa8254762cf910af85e64824e571b79b147887760a`已与远端一致核验。后续需诊断辅助监督贡献并预注册额外配对种子；本次没有自动扩展。


### C8/P10用户授权Test结果（2026-09-07）

用户明确要求“结果以测试集为准”。最终性能汇报以Test macro Dice/IoU为主，Val用于训练选择和机制筛选。此次授权评估C8/P10的原Val IoU选定epoch80检查点，固定阈值0.5，无重训、Test挑检查点或Test调阈值。复用C4历史同协议Test结果；各2113样本，比较按图像配对。原冻结训练manifest的Test禁止标记描述训练阶段，本次独立评估已获得后续用户授权。截至本次C8/P10也已访问Test，不能再称其Test未访问。

| 实验 | Test macro Dice | IoU | Precision | Recall | Brier |
|---|---:|---:|---:|---:|---:|
| C4基线 | 0.842989 | 0.756396 | 0.839363 | 0.874378 | 0.016938 |
| C8仅辅助监督 | 0.844313 | 0.758612 | 0.843207 | 0.873427 | 0.016740 |
| P10辅助监督+路由 | 0.843951 | 0.758090 | 0.832287 | 0.885560 | 0.016846 |

C8-C4：Dice+0.001324，配对图像bootstrap95%CI[-0.000605,0.003290]；IoU+0.002216，CI[-0.000004467,0.004463]；precision+0.003844，CI[0.001823,0.005844]。Test数值有小幅改善，但Dice和IoU区间均跨0，不能声称显著或稳定增益；Val单种子通过仍只是Val结论。

P10-C4：Dice+0.000962，CI[-0.001635,0.003569]；IoU+0.001694，CI[-0.001309,0.004688]；precision-0.007076，CI[-0.009901,-0.004268]。P10-C8：Dice-0.000362、IoU-0.000521，两者区间跨0；precision-0.010921，CI[-0.013471,-0.008407]。完整路由仍未证明增益。额外配对种子与机制归因尚未完成。

两组来源分别为C8 `21606e02c2d64ae950c0f243f55163f58cbedf83`、P10 `2f33a71219ed2beb339db45d702ecc36a931e0c8`；脚本哈希、来源SHA、最佳epoch和样本数均核验。结果目录 `D:/BetterLViT/outputs/race_pe_v2_test_20260907/`，归档SHA256 `01f31f2052ddf4b6eaa6d9dedd1c3183e439ed996783782e11c182cf90e6b9de`与远端一致。status complete，无新训练。

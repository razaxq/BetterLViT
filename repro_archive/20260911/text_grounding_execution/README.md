# 文字创新方向：执行入口

## 当前：A已后台派发，首次检查22:16

悉尼2026-09-11 **22:10:15.563**已提交A，PID **272504**，执行来源`c694e57be6b5ce224489ff9bab2acbb4d4846358`，目录`/root/text_grounding_a_v2_20260911`。启动前RS3四份完成/独立/HF验证证据通过，源文件哈希一致、GPU空闲，系统盘余3,125,993,472字节；后台派发后SSH已断开。见`launch.json`、`state.json`，现在只有提交回执，尚未确认GPU预检通过或得到Val结果。

同一heartbeat `betterlvit` 已改约**今天22:16悉尼时间首次检查**，读回核验完成，检查预算0/2。到时只执行一次`control.py collect`，若仍运行则按实测Train批耗时预测结束并安排唯一末检，若已完成则直接归档。见`first_check_appointment_verified.json`。不要重复deploy或launch，不提前查询GPU。

RS3前置训练/Val及HF归档已全部完成：Val IoU72.8557%、Dice82.5468%，未过预设综合门；末检距训练结束13.20分钟、检查2/2。三组结果已在GitHub提交`4680701`归档，完整结论见[区域监督首轮汇总](../regional_supervision_execution/FIRST_ROUND_REPORT.md)。

## 方案及历史准备记录

用户于2026-09-11授权执行研究报告的分阶段方案。本目录实现第一步A：R2两个直接文字入口的冻结诊断。完整规则、指标及后续分支见[PLAN.md](PLAN.md)，研究依据和先前工作边界见[研究报告](../text_grounding_research/REPORT.md)。

本地已完成规则实现及Train文字静态检查：5716条中5512条可完整解析，204条确定回退；2388条可改变左右关系；所有生成文字≤32 token、无UNK或截断。语义匹配虽然改变4726条原始写法，实际token改变2250条，报告会同时保留这两个数。

GPU实际预检和1429张完整Val按RS3原定22:00悉尼时间末检、归档核验后的顺序接续。以`deployment.json`、`launch.json`、`state.json`和`results/independent_verification.json`依次区分部署、派发、观测、完成核验；不能将准备或派发写成预检通过或已有IoU/Test结果。

执行顺序：RS3收尾 → A文字诊断 → 解释原文/同义/关系干预 → 冻结B的小分支对照方案 → 受控短程筛选。B通过必要条件后才考虑80轮正式比较。

诊断代码的独立完整Git SHA及服务器逐文件校验值保存在部署回执；不修改已冻结的R2或RS3源码、权重和训练配方。现有同一个heartbeat负责接续，执行控制禁止提前启动、重复启动或超过两次检查。

首次部署`f031ad7922042ff3c1b12e6e5b7dcdde3cdb764c`的服务器CPU检查暴露控制脚本对目录深度的假设，8项通过、1项导入失败；未加载模型、未访问GPU或查询训练。失败标签及目录保留，修复后使用独立v2目录及标签，后续只以成功`deployment.json`的来源为准，见`deployment_attempt_v1.json`。

**成功部署已核验：** 2026-09-11 20:49悉尼时间，`c694e57be6b5ce224489ff9bab2acbb4d4846358`已部署到`/root/text_grounding_a_v2_20260911`；10份文件SHA256与Git源完全一致，服务器9项CPU测试通过。GitHub分支及标签`diagnostic-r2-text-path-a-v2-20260911`均指向该完整SHA。未加载模型、未查询GPU或RS3训练。

20:50:59读回核验同一heartbeat `betterlvit`，保留今天22:00的RS3唯一末检；RS3完成归档核验后启动A，随后依实测批耗时改约诊断自身的末检，再按证据进入B。完整工具回执和保存配置见`appointment_verified.json`。当时没有`launch.json`或诊断成绩；最新派发状态见本页顶部。

用户随后明确发出“启动”指令；已将启动请求登记至同一个ACTIVE预约并读回确认，见`queued_start_request.json`。保留22:00的RS3末检及完成核验依赖，满足后直接派发A，无需再次确认。本次仅核对本地文件与预约，未查询RS3/GPU；排入预约队列不等于GPU任务已启动。

# 文字创新方向：执行入口

用户于2026-09-11授权执行研究报告的分阶段方案。本目录实现第一步A：R2两个直接文字入口的冻结诊断。完整规则、指标及后续分支见[PLAN.md](PLAN.md)，研究依据和先前工作边界见[研究报告](../text_grounding_research/REPORT.md)。

本地已完成规则实现及Train文字静态检查：5716条中5512条可完整解析，204条确定回退；2388条可改变左右关系；所有生成文字≤32 token、无UNK或截断。语义匹配虽然改变4726条原始写法，实际token改变2250条，报告会同时保留这两个数。

GPU实际预检和1429张完整Val需在RS3原定22:00悉尼时间末检、归档核验之后运行。准备/部署完成不等于已运行，也没有新IoU/Test结果。以`deployment.json`、`launch.json`、`state.json`和`results/independent_verification.json`依次区分部署、派发、观测、完成核验。

执行顺序：RS3收尾 → A文字诊断 → 解释原文/同义/关系干预 → 冻结B的小分支对照方案 → 受控短程筛选。B通过必要条件后才考虑80轮正式比较。

诊断代码的独立完整Git SHA及服务器逐文件校验值保存在部署回执；不修改已冻结的R2或RS3源码、权重和训练配方。现有同一个heartbeat负责接续，执行控制禁止提前启动、重复启动或超过两次检查。

首次部署`f031ad7922042ff3c1b12e6e5b7dcdde3cdb764c`的服务器CPU检查暴露控制脚本对目录深度的假设，8项通过、1项导入失败；未加载模型、未访问GPU或查询训练。失败标签及目录保留，修复后使用独立v2目录及标签，后续只以成功`deployment.json`的来源为准，见`deployment_attempt_v1.json`。

**成功部署已核验：** 2026-09-11 20:49悉尼时间，`c694e57be6b5ce224489ff9bab2acbb4d4846358`已部署到`/root/text_grounding_a_v2_20260911`；10份文件SHA256与Git源完全一致，服务器9项CPU测试通过。GitHub分支及标签`diagnostic-r2-text-path-a-v2-20260911`均指向该完整SHA。未加载模型、未查询GPU或RS3训练。

20:50:59读回核验同一heartbeat `betterlvit`，保留今天22:00的RS3唯一末检；RS3完成归档核验后启动A，随后依实测批耗时改约诊断自身的末检，再按证据进入B。完整工具回执和保存配置见`appointment_verified.json`。当前没有`launch.json`或诊断成绩，不能写为GPU任务已派发。

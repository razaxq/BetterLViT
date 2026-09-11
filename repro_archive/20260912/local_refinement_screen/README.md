# F：局部纠错接口筛选

**2026-09-12 01:34:44.262悉尼时间全部完成，四组均未通过预注册门，暂不进入文字增量训练。** 来源`a76e1005d5f4b028f5e991c2cf79564b79e66203`，标签`pilot-local-refinement-f2-20260912`，远端`/root/local_refinement_f_a76e1005`。首次检查距全部完成12.008分钟，检查1/2关闭，同一heartbeat已暂停。当前结果属于Train内机制筛选，不能作为正式Test成绩。[完整报告](REPORT.md)。

独立核验15份原始产物、4585图指标与转移、全部训练预算/分组，以及20个头与恢复优化器严格载入通过。HF `a76e1005/` 已核验30文件、48324837字节与全部Xet哈希。保量两组五折均小幅正向，但fine_mass相对R2仅+0.05435 pp且Precision略降；相对coarse_mass仅+0.00099 pp、区间跨零。无保量两组五折均下降，保留全部负结果，不择折或延长。

CPU/CUDA与4585张原R2缓存精确身份、四组真实fit可训练性预检全部通过。源/运行信息见manifest.json、deployment.json、preflight/runtime.json、preflight/benchmark.json和state.json。历史启动时间01:22:42.763，预检每四头一步0.07419秒、曾预测01:41:46完成并预约01:45检查；实际以results/runtime.json为准。前驱标签维度失败与仅清理自身空缓存的证明在attempt_v1，原B缓存未动。

协议见[PROTOCOL.md](PROTOCOL.md)。共4585张原B fit，五折分别849/875/904/937/1020张；四组coarse_free/coarse_mass/fine_free/fine_mass，每组每折2048步，原R2冻结。新增点缓存610355200字节。

## 运行顺序

从本目录执行，或传脚本绝对路径。控制器使用现有SSH密钥，不含凭据。

1. `prepare.py` 生成固定manifest与split；所有源文件提交并打标签。
2. `control.py preflight_launch` 部署已提交字节并后台启动缓存/CPU/CUDA预检。
3. 到deployment.json内预测时间后，`control.py preflight_collect` 一次收取；若失败修复前保留失败来源及缓存状态，不能跳过断言。
4. 仅预检complete、源字节仍一致时，`control.py train_launch` 启动五折四组训练，并生成实测ETA和首次检查预约。
5. 同一原生heartbeat按state.json时间调用 `control.py train_collect`，最多两次；首次完成即关闭。尚在运行则使用state.json新的预约时间更新同一heartbeat，不能循环检查。
6. 全部完成后运行 `verify_results.py` 独立复算OOF结果与全部门，CPU严格载入20个头并核验source/manifest/step和有限性；再备份HF、更新REPORT/台账，提交推送结果。

若细接口通过，只进入文字增量的独立预注册研究；不自动运行完整80轮或Test。如果未通过，报告所有四组及每折结果并停止延长，保留负结果。所有旧A/B/D/E/RS训练已封存，不再查询其进度。B缓存是当前依赖，只读复用。

`test_screen_analysis.py` 六项推进规则测试可在本地无Torch环境运行；`check_refiner.py` 数值与CUDA测试在服务器预检执行。传输可压缩，但落盘的原始产物SHA256必须保持一致。

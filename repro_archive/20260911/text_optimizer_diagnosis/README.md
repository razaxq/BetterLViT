# Train优化器诊断 D

**已完成并封存：** 18案例各512步，2026-09-11 23:25:30.404悉尼时间完成；首次检查距完成7.103分钟，检查1/2即关闭。5份原始产物SHA256、18×512步、32fit逐图计数/摘要独立复算通过，原Adam在0/256/512步精确复现B。18头检查点CPU严格载入、source/manifest/步数/张量有限性均通过。

结论：本配置的coupled L2确实导致T4/image塌缩，AdamW/无衰减恢复学习；恢复后T4固定32fit IoU仍−0.2241/−0.2243 pp，无独立IoU收益证据，不启动80轮或Test。[完整报告](REPORT.md)及[下一步可行性计划](NEXT_PLAN.md)。后续E0/E1尚未执行，不重评原B holdout。

HF `razaxq/BetterLViT/c7080ea8/`已核验16文件、13,007,022字节及全部size/Xet，分类Train-only optimizer diagnostic，非正式Test模型。所有运行已经结束，同一heartbeat已暂停并读回核验，避免每天重查旧run；见hf_upload_verified.json、closure_automation_verified.json。原B缓存仍保留供可行性诊断，只读且未复制。以下为启动历史。

**已于2026-09-11 23:21:43.011悉尼时间后台提交，PID275262。** 执行源`c7080ea82ecaca0e1f33df880d168e3eb7cbc6dd`、标签`diagnostic-text-optimizer-d1-20260911`已推送GitHub，目录`/root/text_optimizer_d_c7080ea8`。7份源文件SHA256匹配，CPU/CUDA行为预检通过，启动前GPU空闲、系统盘余1,313,497,088字节，SSH已断开。当前仅派发回执，尚未确认实际优化进展，无新IoU结果。

比较原Adam+L2、Adam无衰减、AdamW，每种包含原六头，各512步。只使用原eligible fit及32张fit诊断，不再读取B内部holdout或官方Val/Test。依B耗时预估约5分钟，约23:27完成；同一heartbeat已改约**23:30首次检查**，本地保存配置与请求逐字段核验通过，检查0/2。首次若已完成即关闭，不使用第二次，不提前连接。见deployment/launch/state及first_check_appointment回执。

来源与输入哈希在manifest，协议在[PROTOCOL.md](PROTOCOL.md)。运行控制器沿用B的干净Git/源字节/CPU-CUDA预检、后台派发、至多两次预测检查及下载SHA验证；原Adam必须精确复现已保存B在0/256/512步的逐图fit指标。

命令：使用`D:/BetterLViT/.codex_tmp/text_audit_py312/Scripts/python.exe -B -X utf8 control.py deploy|launch|collect|verify`。先提交推送源与独立tag，再deploy与launch各一次；其后仅按state约定时间collect，不提前连接。

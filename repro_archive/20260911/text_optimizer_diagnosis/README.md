# Train优化器诊断 D

**已于2026-09-11 23:21:43.011悉尼时间后台提交，PID275262。** 执行源`c7080ea82ecaca0e1f33df880d168e3eb7cbc6dd`、标签`diagnostic-text-optimizer-d1-20260911`已推送GitHub，目录`/root/text_optimizer_d_c7080ea8`。7份源文件SHA256匹配，CPU/CUDA行为预检通过，启动前GPU空闲、系统盘余1,313,497,088字节，SSH已断开。当前仅派发回执，尚未确认实际优化进展，无新IoU结果。

比较原Adam+L2、Adam无衰减、AdamW，每种包含原六头，各512步。只使用原eligible fit及32张fit诊断，不再读取B内部holdout或官方Val/Test。依B耗时预估约5分钟，约23:27完成；同一heartbeat已改约**23:30首次检查**，本地保存配置与请求逐字段核验通过，检查0/2。首次若已完成即关闭，不使用第二次，不提前连接。见deployment/launch/state及first_check_appointment回执。

来源与输入哈希在manifest，协议在[PROTOCOL.md](PROTOCOL.md)。运行控制器沿用B的干净Git/源字节/CPU-CUDA预检、后台派发、至多两次预测检查及下载SHA验证；原Adam必须精确复现已保存B在0/256/512步的逐图fit指标。

命令：使用`D:/BetterLViT/.codex_tmp/text_audit_py312/Scripts/python.exe -B -X utf8 control.py deploy|launch|collect|verify`。先提交推送源与独立tag，再deploy与launch各一次；其后仅按state约定时间collect，不提前连接。

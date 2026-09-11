# Train优化器诊断 D

当前：协议与代码待提交/预检，尚未启动。比较原Adam+L2、Adam无衰减、AdamW，每种包含原六头，各512步。只使用原eligible fit及32张fit诊断，不再读取B内部holdout或官方Val/Test。

来源与输入哈希在manifest，协议在[PROTOCOL.md](PROTOCOL.md)。运行控制器沿用B的干净Git/源字节/CPU-CUDA预检、后台派发、至多两次预测检查及下载SHA验证；原Adam必须精确复现已保存B在0/256/512步的逐图fit指标。

命令：使用`D:/BetterLViT/.codex_tmp/text_audit_py312/Scripts/python.exe -B -X utf8 control.py deploy|launch|collect|verify`。先提交推送源与独立tag，再deploy与launch各一次；其后仅按state约定时间collect，不提前连接。

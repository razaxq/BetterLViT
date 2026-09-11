# 文字分支B：六组固定特征筛选

**已于2026-09-11 22:59:57.133悉尼时间后台提交：** PID274265，六组共同任务，SSH已断开。启动前系统盘余3,106,045,952字节，源码/预检门通过。当前是派发回执，尚未确认Train缓存和优化进度，没有新IoU成绩。见launch.json、state.json。

同一heartbeat `betterlvit` 已改约**今天23:08首次检查**，本地保存配置与请求逐字段核验；检查0/2。首次若已完成直接收尾；否则按实测步耗时预约唯一末检，不能提前或持续连接。首次预约及核验文件以first_check_appointment开头。以下为部署预检历史。

当前：**服务器CPU和CUDA行为预检均通过，尚未提交训练，没有新模型IoU结果。** 每组75,808个可训练参数，CUDA预检最大逐图软面积误差0.000134241像素（注册上限0.02）；finite difference、identity、确定性反传、各对照输入隔离均通过。14份部署文件逐字节SHA256与Git一致，系统盘可用3,106,050,048字节。详见deployment.json。A已完成并独立核验，见[报告](../text_grounding_execution_v3/REPORT.md)。

执行源`49905dbbdc644454a37a4a49098db0d5df5fe75a`，组标签`pilot-r2-text-residual-b-v1-20260911`，均已推送GitHub。服务器目录`/root/text_head_b_49905dbb`。部署等待包括源文件传输；回执最终成功，没有重复部署。

协议见[PROTOCOL.md](PROTOCOL.md)，机器可读参数与5716条固定成员见manifest.json、split.json。六组T1/T2/T3/T4/image/template均1024步，只训练小残差头，R2冻结；458张eligible内部holdout，另报告全部1131张。禁止官方Val/Test访问，结果不等于独立泛化或第二创新成立。

本地运行环境：`D:/BetterLViT/.codex_tmp/text_audit_py312/Scripts/python.exe -B -X utf8 control.py deploy|launch|collect|verify`。部署先要求干净提交、固定全部执行文件字节并在服务器CPU/CUDA预检；服务器目录由部署SHA前8位生成。仅预检通过后执行launch一次并断开SSH。首次预约在8分钟后，最多两次预测式检查；不要在预约前运行collect。

每个阶段的deployment/launch/state/inspection及results为真实状态依据。原始结果以二进制字节保留，summary由逐图TP/FP/FN独立复算。缓存位于独立系统盘目录，原始1,756,686,848字节，总预算1.9GB且留至少1GB空闲；不写shared fs。

重现Train划分可在包含A/B两目录的docs工作树运行`register.py --train-folder PATH/Train_Folder`。它只读取Train成员名及Train成员文字，不读取mask像素；运行前后结果应与已注册split逐字节相同。R2原模型及离线CXR-BERT仍须按manifest准备。

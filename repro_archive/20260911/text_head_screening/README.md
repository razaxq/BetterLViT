# 文字分支B：六组固定特征筛选

**已完成：** 六组于23:04:28.152完成1024步，23:04:50.227完成内部留出评估；首次检查距训练/评估结束4.755/4.387分钟，检查1/2关闭，不再查询本run。原始结果及摘要独立复算通过，基线/缓存/文字身份核验通过，无官方Val/Test访问。全部1131张内部holdout：R2 IoU79.6444%、Dice87.9395%；T1/T2/T3/T4/image/template的IoU变化分别−0.1532/−0.0335/−0.0648/0/0/−0.0132 pp，均无增益，T3/T4门失败。见[完整报告](REPORT.md)。

T4/image输出塌缩，后验只读Train梯度/权重诊断支持coupled L2压制弱分支的解释，尚待优化器对照确认。小头可加载且源码/哈希匹配。HF Bucket `razaxq/BetterLViT/49905dbb/`已核验19文件、15,627,795字节及全部Xet哈希，明确为Train内部机制筛选；见hf_upload_verified.json。接续[Train优化器诊断](../text_optimizer_diagnosis/PROTOCOL.md)，不重评本holdout、不自动开80轮。以下为已完成运行的派发/预检历史。

**已于2026-09-11 22:59:57.133悉尼时间后台提交：** PID274265，六组共同任务，SSH已断开。启动前系统盘余3,106,045,952字节，源码/预检门通过。当前是派发回执，尚未确认Train缓存和优化进度，没有新IoU成绩。见launch.json、state.json。

同一heartbeat `betterlvit` 已改约**今天23:08首次检查**，本地保存配置与请求逐字段核验；检查0/2。首次若已完成直接收尾；否则按实测步耗时预约唯一末检，不能提前或持续连接。首次预约及核验文件以first_check_appointment开头。以下为部署预检历史。

部署时：服务器CPU和CUDA行为预检均通过。每组75,808个可训练参数，CUDA预检最大逐图软面积误差0.000134241像素（注册上限0.02）；finite difference、identity、确定性反传、各对照输入隔离均通过。14份部署文件逐字节SHA256与Git一致，当时系统盘可用3,106,050,048字节。详见deployment.json。A已完成并独立核验，见[报告](../text_grounding_execution_v3/REPORT.md)。

执行源`49905dbbdc644454a37a4a49098db0d5df5fe75a`，组标签`pilot-r2-text-residual-b-v1-20260911`，均已推送GitHub。服务器目录`/root/text_head_b_49905dbb`。部署等待包括源文件传输；回执最终成功，没有重复部署。

协议见[PROTOCOL.md](PROTOCOL.md)，机器可读参数与5716条固定成员见manifest.json、split.json。六组T1/T2/T3/T4/image/template均1024步，只训练小残差头，R2冻结；458张eligible内部holdout，另报告全部1131张。禁止官方Val/Test访问，结果不等于独立泛化或第二创新成立。

本地运行环境：`D:/BetterLViT/.codex_tmp/text_audit_py312/Scripts/python.exe -B -X utf8 control.py deploy|launch|collect|verify`。部署先要求干净提交、固定全部执行文件字节并在服务器CPU/CUDA预检；服务器目录由部署SHA前8位生成。仅预检通过后执行launch一次并断开SSH。首次预约在8分钟后，最多两次预测式检查；不要在预约前运行collect。

每个阶段的deployment/launch/state/inspection及results为真实状态依据。原始结果以二进制字节保留，summary由逐图TP/FP/FN独立复算。缓存位于独立系统盘目录，原始1,756,686,848字节，总预算1.9GB且留至少1GB空闲；不写shared fs。

重现Train划分可在包含A/B两目录的docs工作树运行`register.py --train-folder PATH/Train_Folder`。它只读取Train成员名及Train成员文字，不读取mask像素；运行前后结果应与已注册split逐字节相同。R2原模型及离线CXR-BERT仍须按manifest准备。

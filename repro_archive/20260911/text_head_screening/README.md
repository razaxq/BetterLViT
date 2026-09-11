# 文字分支B：六组固定特征筛选

当前：代码与预注册完成，等待提交后服务器CPU/CUDA行为预检。**尚未启动训练，没有新模型IoU结果。** A已完成并独立核验，见[报告](../text_grounding_execution_v3/REPORT.md)。

协议见[PROTOCOL.md](PROTOCOL.md)，机器可读参数与5716条固定成员见manifest.json、split.json。六组T1/T2/T3/T4/image/template均1024步，只训练小残差头，R2冻结；458张eligible内部holdout，另报告全部1131张。禁止官方Val/Test访问，结果不等于独立泛化或第二创新成立。

本地运行环境：`D:/BetterLViT/.codex_tmp/text_audit_py312/Scripts/python.exe -B -X utf8 control.py deploy|launch|collect|verify`。部署先要求干净提交、固定全部执行文件字节并在服务器CPU/CUDA预检；服务器目录由部署SHA前8位生成。仅预检通过后执行launch一次并断开SSH。首次预约在8分钟后，最多两次预测式检查；不要在预约前运行collect。

每个阶段的deployment/launch/state/inspection及results为真实状态依据。原始结果以二进制字节保留，summary由逐图TP/FP/FN独立复算。缓存位于独立系统盘目录，原始1,756,686,848字节，总预算1.9GB且留至少1GB空闲；不写shared fs。

重现Train划分可在包含A/B两目录的docs工作树运行`register.py --train-folder PATH/Train_Folder`。它只读取Train成员名及Train成员文字，不读取mask像素；运行前后结果应与已注册split逐字节相同。R2原模型及离线CXR-BERT仍须按manifest准备。

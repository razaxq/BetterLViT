# 文字诊断A：文件名接口修复v3

v2来源`c694e57be6b5ce224489ff9bab2acbb4d4846358`在第一个Train批次因文字表键名与DataLoader返回名称不一致停止，没有进入Val或更新模型。失败版本及首次检查封存在[原执行目录](../text_grounding_execution/README.md)，本目录是独立修复版，不覆盖旧运行。

R2约定为`mask_filename.replace('mask_', '')`得到图像名。v3显式保存图像名→mask键→原文字的一一映射，拒绝冲突或缺失；分词ID逐项与冻结数据集缓存核对后再推理。固定Train子集与文字变换仍使用mask键作为身份（与原Train语料检查一致），逐图指标仍使用图像名，与原R2 exporter完全对应。

原`text_policy.py`字节SHA256、所有5716条Train生成变换的SHA256均作为约束保留；不得为获得更好的Val改变文字规则、权重、阈值或样本。原32张Train→1429张Val、15组通路干预、两次预测检查、A后再研究B的顺序保持，见[PLAN.md](PLAN.md)。

运行脚本：`D:/BetterLViT/.codex_tmp/text_audit_py312/Scripts/python.exe -B -X utf8 control.py deploy|launch|collect|verify`。源代码必须先独立提交推送、打标签再deploy；服务器目录为`/root/text_grounding_a_v3_20260911`。以本目录的deployment/launch/state/inspection回执判断阶段，不读取v2的state当成v3状态。

本地11项行为测试已通过，包括本次失败样本的名称映射、缺失/冲突拒绝、原语义控制及通路恢复。全部5716个训练样本的图像—文字对应检查通过，5716个文字键均带mask前缀；生成变换完整SHA256仍为`324cb826d4675e6863dbdb1556a5ce9c326b39d93a450ff62b6e934ead1cce06`，与原语料检查完全一致。此时尚未部署或重启GPU诊断，无新IoU/Test结果；以之后的部署与派发回执为准。

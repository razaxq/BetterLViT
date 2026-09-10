# R2、S1、S2：用户指定的测试集比较

2026-09-11用户明确要求“给我R2,S1,S2的测试集结果”。本次授权新增S1/S2的Test评估，保留原Val筛选失败事实；不将其改写成通过多种子筛选，不改变历史计划或训练源码。

固定seed1219、80轮训练的Val IoU最佳检查点，阈值0.5、batch16，完整2113张Test，逐图macro IoU/Dice/Precision/Recall。R2复用已核验同种子Test JSON（SHA在authorization.json）；S1 Best67、S2 Best75。无重训、无阈值扫描、无检查点重选。

评估器从R2历史评估器派生，只调整授权与架构校验、结果溯源字段；模型构建、Test loader、指标函数和完整推理循环AST一致。S1/S2和R2的ValGenerator、ImageToImage2D、read_text AST亦一致，见evaluator_verification.json。

先提交并推送评估代码，再由control.py launch短连接提交run_chain.py；SSH随提交结束，按历史每组约147秒预测，两组预留320秒，提交后390秒执行一次control.py collect。未启动训练，历史训练检查次数不变。结果须等两个评估进程成功、JSON完整且逐图复算通过后报告。

标签几何修复目前只解决实现一致性；没有证据支持明显IoU增益。本次评估原始S1/S2模型，不混入修正原型。

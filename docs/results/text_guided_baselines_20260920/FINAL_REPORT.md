# 三个文字引导对比方法：已完成

MMI-UNet、GuideDecoder/LanGuideMedSeg、DD-CMD 的同八图预测全部齐全，图像和掩膜按原文件名保存。图库：http://127.0.0.1:8769/three_methods/index.html 。目录：outputs/external_methods_20260920/three_methods，下载包three_methods.zip，指标metrics.csv。你的FSDR＋完整RACE使用既有单次余弦seed2027预测。

|方法|seed|训练轮数|Best轮|Val IoU|Val Dice|Test IoU|Test Dice|
|---|---:|---:|---:|---:|---:|---:|---:|
|GuideDecoder|1219|100|26|72.0985%|81.8530%|75.2936%|83.8836%|
|DD-CMD|1219|160|115|73.5672%|83.0536%|76.2994%|84.7533%|
|MMI-UNet|作者权重|未重训|不适用|未评估|未评估|75.4783%|83.9707%|

新增两项同项目Train5716/Val1429/Test2113，官方结构与各自主要配方：224px、24tokens、冻结对应BERT、无LoRA、AdamW wd0.01、FP32、官方损失与增强、单周期余弦Tmax200最低1e-6；Guide为CXR-BERT/ConvNeXt、batch16、lr3e-4、原概率输出DiceCELoss，DD为BiomedBERT/ConvNeXt、batch8、lr5e-5、logits DiceCELoss(sigmoid=True)。Val逐图macro IoU选择Best，固定Best一次Test，阈值严格>0.5。全部模型、配置、SHA来源详见PLAN.md、manifest.json和FINAL_RESULTS.json。

核验完成：服务器结果文件SHA256、两方法2113 Test样本集合一致、逐图均值重算、Best权重与源码来源、八图原GT方向及掩膜/概率/指标一致。两页误差总图已人工视觉检查，ZIP完整性及图库HTTP200均通过。原有权重、日志和旧图库保留；不新增推理或训练。

八图为用户此前选定病例，不代表总体增益。MMI作者权重的训练文件清单未独立核验；新增两方法为本项目划分重训，配方不同，不能混称严格官方论文成绩复现。

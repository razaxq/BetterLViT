# 局部重编码可行性实验：执行证据

2026-09-10 15:11:02.919（悉尼）完整完成两组2400步及Val导出。用户15:23:24.172触发唯一末检，距结束741.254秒，检查1/2，符合30分钟要求。全部产物已下载并核验，没有持续SSH。两组均无有效IoU增益，不通过推进门，详见[完整结果](../results/README.md)。

原始启动于14:56:18，PID200703；原预测15:19完成、heartbeat `betterlvit`预约15:23。末检时该定时已不存在，删除接口返回`not_found`，原因未知；本次实际检查由用户触发，不宣称自动预约成功执行。

源码 `c749b72db327cd777f63a4b6da4583b301456538`，tag `diagnostic-r2-local-reencode-20260910`；两者均已推送GitHub `razaxq/BetterLViT`。基线R2来源 `9eca26de5b301099805530edbf5a1a8718bea662`。独立GPU预检两次8步完全一致、每组192,129参数、各模块均有非零梯度，初始化输出恒等、冻结状态不变、区域预算与外部不变性通过。两组步骤合计约0.29346秒，显存峰值5,453,965,312字节。

共享fs实际18,559,782,256字节，低于20,000,000,000硬上限；启动时系统可用3,989,245,952字节。本轮仅缓存内存特征，不写大型特征文件、不复制基线权重。

- `preflight.json`：预检、时间估计、来源与脚本哈希。
- `launch.json`：后台提交回执和逐文件冻结哈希。
- `state_at_launch.json`：原始时间预测与检查预算。
- `automation.json`：定时预约快照。
- `inspection_1.json`、`state.json`：唯一末检原始证据与完成状态。
- `automation_cleanup.json`：定时不存在的接口确认。

末检通过`control.py collect`核验完整Val、原始R2逐图指标与所有文件哈希；`finalize.py`进一步独立重建8425行计数指标、验证训练/审计/Val文件名互斥及2400步/LR端点。清单的audit_names为mask文件名，按原数据加载器的mask_去除规则映射到image文件名后核对；训练原始清单未修改。

结果完成、归档和提交后已运行`upload_completed.py`。27文件、5,748,879逻辑字节已新增至HF Bucket `razaxq/BetterLViT` 的 `c749b72d/local_reencode_v1/`，覆盖两组分支权重、逐图结果、冻结源码和执行材料；逐文件大小及Xet哈希一致，证明见`hf_upload_verified.json`。既有模型未覆盖或删除。本轮没有访问Test。

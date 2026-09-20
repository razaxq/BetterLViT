# FSDR — Frequency-aware Semantic–Detail Refinement

中文正式名称：**频率感知的语义—细节精炼模块**。

## 正式命名决定（2026-09-14）

经用户确认，BetterLViT 当前采用的 **FAM-EPPA V4-B** 正式更名为 **FSDR**。
此后论文、答辩、图表及新文档中的正式模块名称统一使用 FSDR。
这是名称更新，计算结构、参数、训练协议及已有实验结果均不因更名而改变。

首次出现建议写作：

> Frequency-aware Semantic–Detail Refinement (FSDR)

中文首次出现建议写作：

> 频率感知的语义—细节精炼模块（Frequency-aware Semantic–Detail Refinement，FSDR）

需要关联旧记录时写作 **FSDR（原 FAM-EPPA V4-B）**。
正式展示名称不再叠加 FAM 前缀或 V4-B 开发版本后缀；它们保留在历史对照说明中。
早期 EPPA、DG-EPPA、V4-A、V4-C 等不同结构仍使用各自历史名称，不能仅因共享 EPPA 字样就视为同一 FSDR 模型。

## 模块含义与结构

- **Frequency-aware**：通过固定 Haar 分解重建低频与高频分量，并在 `up4`、`up3` 使用空间自适应的低通/高通滤波混合。
- **Semantic–Detail**：低频路径融合 CNN skip、ViT 重建引导及解码器语义，结合文本条件和特征一致性生成语义支持；高频路径保留并增强细节。
- **Refinement**：以受限残差更新跳跃连接特征，再与上采样的解码器特征融合。

一句话介绍：将特征分解为低频语义与高频细节，融合多源语义信息，并利用语义支持增强细节，以精炼跳跃连接特征。

实现中的 `plam` / `plam1..plam4` 在该结构中指 **ViT 重建引导特征**，不应解释为原版 PixLevelModule 的输出。

## 代码入口与复现兼容

新代码推荐使用：

```python
from nets.fsdr import FSDR
```

`FSDR` 与原 `nets.eppa.EPPA`、`nets.eppa.FAMAdaptiveHaarEPPA` 指向同一个类，不增加包装层、不新增参数。
LViT 的构造入口使用 FSDR，但注册的子模块名仍为 `eppa`，保留原权重键。

以下技术标识继续保留，以支持旧配置、检查点、脚本和归档：

| 标识 | 保留原因 |
|---|---|
| `nets/eppa.py`、`EPPA`、`FAMAdaptiveHaarEPPA` | 保持旧导入路径与类引用可用 |
| `fam_eppa_v4b`、`paper_*_fam_eppa_v4b_*` | 保持架构校验与实验配置匹配 |
| `eppa_*` 配置键、`up*.eppa.*` 权重键 | 保持配置行为与检查点加载一致 |
| 历史 Git 分支、标签、完整 SHA、运行目录、结果 JSON | 保留实验溯源事实 |

既有实验表可标注“FSDR（原 FAM-EPPA V4-B）”，但不得将更名描述为新增实验或新的性能提升。
更名不改变 LoRA/无 LoRA、Val/Test、macro/micro、阈值及训练配方的比较边界。

## 关联资料

- [实验台账](EXPERIMENT_TRACKER.md)
- [V4-B 历史设计与实验协议](FAM_EPPA_V4B_EXPERIMENT.md)
- [早期 EPPA 历史说明](EPPA.md)（不是当前 FSDR 的结构规范）
- [正式代码入口](../nets/fsdr.py)
- [兼容实现](../nets/eppa.py)

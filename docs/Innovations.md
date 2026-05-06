# BetterLViT — 相对于 vanilla LViT 的全部创新与改造

> 文档目的: 给毕业论文写作直接引用. 涵盖 `eppa` 分支相对 `main`(vanilla LViT) 的所有架构改造、工程改进、超参数变化,
> 包含每一项的动机、实现位置、设计取舍.
> 所属分支: `eppa`
> 基线分支: `main` (vanilla LViT 的本地 fork)
> 文件覆盖: `Config.py`, `Load_Dataset.py`, `Train_one_epoch.py`, `nets/BetterLViT.py`, `nets/LViT.py`, `nets/eppa.py`,
`nets/Vit.py`, `test_model.py`, `train_model.py`, `utils.py`

---

## 0. 速览表

| # | 创新                                                  | 类别 | 关键文件                                    | 状态                                                                    |
|---|-----------------------------------------------------|----|-----------------------------------------|-----------------------------------------------------------------------|
| 1 | **CXR-BERT 文本编码器 + LoRA** 替换原 `bert_embedding`      | 架构 | `nets/BetterLViT.py`, `Load_Dataset.py` | ✅ 已生效                                                                 |
| 2 | **FreqEPPA — 频率分解式注意力** 替换原 `PixLevelModule (PLAM)` | 架构 | `nets/eppa.py`, `nets/LViT.py`          | ✅ 已生效                                                                 |
| 3 | 复合边界感知损失 `WeightedTverskyFocalBoundary`             | 监督 | `utils.py`                              | ⚠ **存在于 `eppa-boundary` 分支, 当前 `eppa` 分支未接入**; 训练仍用 `WeightedDiceBCE` |
| 4 | 训练断点恢复 (resume from checkpoint)                     | 工程 | `train_model.py`, `Config.py`           | ✅ 已生效                                                                 |
| 5 | 滚动 last_model 保存 + epoch_history 持久化 + best 模型自带历史  | 工程 | `train_model.py`                        | ✅ 已生效                                                                 |
| 6 | EPPA gate 诊断 (gate-history 子表)                      | 诊断 | `train_model.py`                        | ✅ 已生效, FreqEPPA 上自动跳过                                                 |
| 7 | Bark 推送通知 (训练开始/最佳/结束) + 自动关机                       | 工程 | `train_model.py`                        | ✅ 已生效                                                                 |
| 8 | thop 依赖剥离 (与 PEFT 不兼容)                              | 工程 | `train_model.py`                        | ✅ 已生效                                                                 |
| 9 | 训练超参数微调 (epochs, weight_decay 等)                    | 超参 | `Config.py`                             | ✅ 已生效                                                                 |

---

## 1. 创新 1 — CXR-BERT 文本编码器 + LoRA

### 1.1 动机

vanilla LViT 使用一个静态的 `bert_embedding` 文件 (硬编码 token-id → 768d 词向量), 不通过 BERT 模型本体编码语境.
这种方式存在两个问题:

1. **缺乏临床上下文**. 医学放射报告的语境信息 (如 "consolidation in the left lower lobe") 是顺序敏感的, 静态词向量无法捕捉.
2. **领域漂移**. `bert-base-uncased` 在通用语料上预训练, 与胸部 X 光报告的医学专业语料分布差异显著.

### 1.2 解决方案

**编码器替换**: 改为微软 `microsoft/BiomedVLP-CXR-BERT-specialized` —— 在大规模 chest X-ray 报告语料上经过 image-text
对比学习预训练的 BERT 变体, 其 `[CLS]` token 与图像表征对齐, 并保证 token-level 表征不被 `[PAD]` 污染.

**参数高效微调 (PEFT/LoRA)**: 用 PEFT 库的 LoRA 适配器, 在 BERT 每个 transformer block 的 **6 个 Linear**
上插入低秩适应矩阵: `query`, `key`, `value`, `attention.output.dense` (attention 输出投影), `intermediate.dense` (FFN
expand 768→3072), `output.dense` (FFN contract 3072→768)。注意 PEFT 的 `target_modules` 用后缀匹配, 所以 `'output.dense'`
同时命中 attention 输出和 FFN 输出两类 Linear, 配置仅需 5 个字符串。BERT 主干冻结, 仅更新 LoRA 矩阵。当前超参
`r=16, α=32, dropout=0.1`, 总可训参数约 **2.65M (占 BERT ~2.4%)** vs 全微调的 110M。

> **配置选择的物理理由**: 下游消费链是 `BERT → text[:, 0, :] → EPPA.text_proj`. [CLS] 表征在 BERT 内部经过 12 个
> transformer block, 每个 block 的 FFN 都把 [CLS] 重新投影一次. 仅 q+v 适配能让 [CLS] 在 attention 阶段聚合不同 token, 但
> 12 个**冻结**的 FFN 把聚合结果强行推回预训练分布——CXR-BERT 预训练把 [CLS] 推到 image-text contrastive 对齐空间, 但
> segmentation 下游需要的是"病灶在哪类通道"的语义。FFN LoRA 把 12 段冻结流水线变成轻度可调, 是该任务的关键 ROI 来源 (而 K
> 单独加只有边际收益, 主要靠与 FFN 协同发挥作用).

### 1.3 实现位置

```python
# nets/BetterLViT.py:43-59
self.text_encoder = AutoModel.from_pretrained(
    text_encoder_name, trust_remote_code=True)

if use_lora:
    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha,
        target_modules=list(lora_target_modules),  # 来自 Config.text_lora_target_modules
        lora_dropout=lora_dropout, bias='none')
    self.text_encoder = get_peft_model(self.text_encoder, lora_cfg)
else:
    for p in self.text_encoder.parameters():
        p.requires_grad = False
```

```python
# Config.py:46-62  当前默认 (扩展全 6 模块)
text_lora_target_modules = (
    'query', 'key', 'value',
    'intermediate.dense', 'output.dense',  # 后缀匹配同时命中 attn.output 和 FFN.output
)
# 切回 ('query', 'value') 即复现旧的 q+v 消融组
```

数据管线 (`Load_Dataset.py`) 同步换成 HuggingFace `AutoTokenizer`, 输出 `{input_ids, attention_mask}` 二元组, 序列长度由
`Config.text_max_len` 控制 (默认 32, 较 vanilla LViT 的硬编码 10 更长).

### 1.4 关键约束 (写论文要 cite 的不变量)

- **隐藏维度 hardcoded 为 768**: `LViT.text_module4 = Conv1d(768, 512, ...)`. 任何文本编码器替换必须输出 768d, 否则需同步修改
  conv stack.
- **序列长度通过 `text_seq_len` 串通**:
  `Config.text_max_len → BetterLViT → LViT → VisionTransformer → CTBN3.in_channels`. 改 `text_max_len` 必须同步, 否则
  `CTBN3.Conv1d` 形状不对会静默错位.
- **`[CLS]` 用于通道注意力的语义条件**: 不切到 mean-pool, 因 CXR-BERT-specialized 的 `[CLS]` 在预训练时与图像对比对齐,
  且天然不含 `[PAD]` 污染.

### 1.5 消融配置

`Config.model_name` 提供三态:

- `'BetterLViT'`: CXR-BERT + LoRA (主创新组)
- `'LViT'`: CXR-BERT 全冻结 (无 LoRA, 仅看预训练表征本身的贡献)
- `'LViT_pretrain'`: 加载 MoNuSeg 预训练 UNet 权重作为 warm-start

### 1.6 论文可直接引用的段落 (中英双版)

> **EN**: We replace LViT's static `bert_embedding` lookup with the contextual CXR-BERT-specialized encoder (Boecking et
> al., 2022), pretrained on chest X-ray report corpora with image-text contrastive alignment. To avoid catastrophic
> forgetting and reduce trainable parameters, we apply LoRA (Hu et al., 2021) with rank 8, scaling factor 16, and dropout
> 0.15 on the `query` and `value` projections of all self-attention layers; the BERT backbone is frozen. The encoder
> output (`[B, 32, 768]`) feeds into LViT's existing text-projection cascade unchanged.

> **CN**: 我们用 CXR-BERT-specialized (Boecking 等, 2022) 替换 LViT 的静态 `bert_embedding` 词表查找; 该编码器在胸部 X
> 光报告语料上经过 image-text 对比预训练, 提供医学语境感知的 token 序列表征. 为避免灾难性遗忘并减少可训参数量, 我们对
> self-attention 的 `query` 和 `value` 投影添加 LoRA 适配器 (rank=8, scaling=16, dropout=0.15), 冻结 BERT 主干. 编码器输出
`[B, 32, 768]` 直接接入 LViT 原有的 text-projection 级联, 模型架构其余部分不变.

---

## 2. 创新 2 — FreqEPPA: 频率分解式跳跃注意力

### 2.1 演进史 (论文 Discussion 章节的反思素材)

我们在 `eppa` 分支上**经历了三次架构迭代**, 每次都由实证诊断驱动:

| 版本                   | 设计                                    | 失败原因                                             | 诊断文件                 |
|----------------------|---------------------------------------|--------------------------------------------------|----------------------|
| EPPA v1 (CBAM-style) | `x + gate · x · ca · sa`, gate init=0 | gate 死锁: abs_mean<0.01 且向 0 收缩, mean 系统性偏负       | git commit `e58c67d` |
| EPPA v2 (gate=0.1)   | 同 v1, gate init=0.1                   | gate 仍指数级衰减 (~ -22%/epoch), 11 epoch 内回到 v1 死亡水平 | git commit `b367837` |
| FreqEPPA (当前)        | 频率分解 + 路由, 无 gate                     | 见 §2.4 设计鉴别                                      | git commit `3267f13` |

诊断手段: 每 epoch 在 `train_model.py` 训练日志中追加 `EPPA Gate History` 子表, 记录 `mean / abs_mean / max / min` (实现
commit `a086e0f`).

### 2.2 根因分析 (论文 Method 章节的关键 framing)

EPPA v1/v2 的核心架构是 CBAM-style:

$$x_{\text{sa}} = x \cdot \text{ca} \cdot \text{sa}, \quad \text{ca}, \text{sa} \in (0, 1)$$

$$\text{output} = x + g \cdot x_{\text{sa}}$$

由于 ca, sa 经 sigmoid 都 ∈ (0, 1), 所以 $|x_{\text{sa}}| \leq |x|$ 元素级恒成立. 残差只能**衰减** x, 无法**注入** x
中没有的边界信息. 在 dice ≈ 0.83 的饱和状态下, 剩余误差几乎都在边界 1–2 像素的模糊区, 而模糊区的特点正是 x 在那里接近 0;
衰减 x 在最该出力的位置信号最弱. Adam 的最优化策略最终选择 "关闭 EPPA 分支" (gate → 0).

### 2.3 FreqEPPA 架构

```
input x
  │
  ├── self.low_pass (Depthwise Conv2d 3×3, Gaussian-init) ───── x_low
  │                                                               │
  │    ┌─ avg(spatial), max(spatial) ─┐                            │
  │    └────── ch_mlp + text_proj ─────┴── ch_logit                │
  │                                          │                    │
  │                                  ca = 1 + 0.5·tanh(·)         │
  │                                  范围 (0.5, 1.5)               │
  │                                                                │
  └── x - x_low ── x_high ── |·| ── avg(C), max(C) ── 3×3 conv     │
                                          │                       │
                                  sa = 1 + tanh(·)                │
                                  范围 (0, 2)                      │
                                                                   │
                              ┌─────────────────────────────────────┘
                              ▼
                output = x_low · ca + x_high · sa
```

四个关键设计决策:

| 决策    | 选择                                     | 替代方案                                    | 决策依据                                                     |
|-------|----------------------------------------|-----------------------------------------|----------------------------------------------------------|
| 低通滤波器 | depthwise 3×3 conv, Gaussian-init, 可训练 | 固定 Gaussian buffer (不训练) / `avg_pool2d` | 论文中可论述 "per-channel cutoff 自适应"; 可训练但 Gaussian-init 给定先验 |
| 高通构造  | `x - x_low` (减法, 零参数)                  | 另一个独立可训卷积                               | 保证 `x_low + x_high ≡ x` 严格相等, identity-at-init 才数学严格成立   |
| ca 范围 | `(0.5, 1.5)` 通过 `1 + 0.5·tanh`         | `(0, 1)` 通过 sigmoid                     | 允许通道**放大** (>1) 而非只能衰减                                   |
| sa 范围 | `(0, 2)` 通过 `1 + tanh`                 | `(0, 1)` 通过 sigmoid                     | **关键**: 允许 sa>1 实现 unsharp masking 风格的**边缘锐化**           |

### 2.4 Identity-at-init 论证

论文里要严格证明 init 时输出 = input:

```
ch_mlp[-1].weight = 0  →  ch_logit = 0  →  ca = 1 + 0.5·tanh(0) = 1
text_proj.weight,bias = 0  →  text 贡献 = 0
sp_proj.weight = 0  →  sp_logit = 0  →  sa = 1 + tanh(0) = 1
low_pass.weight = Gaussian (sums to 1)  →  x_low ≈ x_blurred
高通: x_high = x - x_low  (恒等保持)

output = x_low · 1 + x_high · 1
       = x_low + (x - x_low)
       = x   ←  严格相等 (与 Gaussian 的频率响应无关)
```

**这是 FreqEPPA 取代 LayerScale gate 的关键**: identity-at-init 不再依赖一个会被 Adam 关掉的可学习标量, 而是**结构性**
保证. zero-init 最后一层的梯度仍非零 (∂loss/∂W_last ∝ upstream · input), 所以训练从 step 1 开始即可推动 W_last 离开零,
进而解锁前层. 此为 DiT (Peebles & Xie 2023) 与 ControlNet (Zhang et al. 2023) 的标准 safe-init 技巧.

### 2.5 实现位置

完整实现见 `nets/eppa.py:6-134`. 关键代码片段:

```python
# nets/eppa.py:75-85 — 低通卷积 + Gaussian init
self.low_pass = nn.Conv2d(
    in_channels, in_channels, kernel_size=3, padding=1,
    groups=in_channels, bias=False)
gaussian = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.0
with torch.no_grad():
    self.low_pass.weight.copy_(gaussian.expand(in_channels, 1, 3, 3))


# nets/eppa.py:110-134 — forward
def forward(self, x, text=None):
    x_low = self.low_pass(x)
    x_high = x - x_low
    # ... ch_logit, ca ∈ (0.5, 1.5)
    # ... sp_logit, sa ∈ (0, 2)
    return x_low * ca + x_high * sa
```

### 2.6 复杂度分析 (论文要给出参数量)

每个 EPPA 实例 (in_channels = C):

| 组件                     | 参数量                             | 说明                                     |
|------------------------|---------------------------------|----------------------------------------|
| `low_pass`             | `9C`                            | depthwise, 9 kernel slots × C channels |
| `ch_mlp` (in→c_red→in) | `C·c_red + c_red·C = 2·C·c_red` | bias=False; `c_red = max(C/8, 32)`     |
| `text_proj`            | `C·768 + C`                     | weight + bias                          |
| `sp_proj`              | `2·9 = 18`                      | bias=False                             |
| 总计                     | `~9C + 2C·c_red + 769C + 18`    |                                        |

四个 stage (C=512, 256, 128, 64) 总参数量约 **1.5M**, 远低于 ViT cross-attention 的开销.

### 2.7 论文可直接引用的段落

> **EN**: We propose Frequency-Routed EPPA, which decomposes the skip features via a depthwise low-pass convolution (
> initialised as a 3×3 Gaussian kernel) into low- and high-frequency components ($x_{\text{low}}$
> and $x_{\text{high}} = x - x_{\text{low}}$), then routes them to separate attention branches: low-frequency drives a
> text-conditioned channel attention ($ca = 1 + 0.5 \tanh \in (0.5, 1.5)$) capturing semantic context, while
> high-frequency drives a spatial attention on the absolute response ($sa = 1 + \tanh \in (0, 2)$) emphasising boundary
> localisation. The recomposed output $x_{\text{low}} \cdot ca + x_{\text{high}} \cdot sa$ permits unsharp-masking-style
> edge sharpening when $sa > 1$, structurally breaking the attenuation-only constraint that sank our earlier CBAM-style
> EPPA variants. Identity at initialisation is achieved structurally via zero-init of the final linear / convolutional
> layers (the DiT/ControlNet safe-init pattern), removing the need for a LayerScale gate.

> **CN**: 我们提出 Frequency-Routed EPPA: 用初始化为 3×3 Gaussian 的 depthwise 卷积将跳跃特征分解为低频 $x_{\text{low}}$
> 与高频残量 $x_{\text{high}} = x - x_{\text{low}}$, 路由到独立的两个注意力分支:
> 低频驱动文本条件化的通道注意力 ($ca = 1 + 0.5\tanh \in (0.5, 1.5)$) 捕捉语义区域信息,
> 高频驱动作用于幅值响应的空间注意力 ($sa = 1 + \tanh \in (0, 2)$) 聚焦边界定位.
> 输出 $x_{\text{low}} \cdot ca + x_{\text{high}} \cdot sa$ 在 $sa > 1$ 时实现 unsharp-masking 风格的边缘锐化, 结构性地突破了
> CBAM 衰减式注意力 $|x_{\text{sa}}| \leq |x|$ 的限制 (此限制是我们前两版 EPPA 失败的根因). identity-at-init 由各最终层
> zero-init 结构性保证 (即 DiT/ControlNet 的 safe-init 范式), 不再需要 LayerScale 门控.

---

## 3. (待办) 创新 3 候选 — 复合边界感知损失

### 3.1 当前状态

⚠ **该创新存在于 `eppa-boundary` 分支但未合入当前 `eppa` 分支**. 训练循环 (`train_model.py:184`) 当前仍使用基线损失
`WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)`.

### 3.2 设计 (来自 `eppa-boundary` 分支的提交记录)

| 组件                | 公式/参数                                                    | 作用                      |
|-------------------|----------------------------------------------------------|-------------------------|
| Tversky loss      | $\alpha = 0.45$, $\beta = 0.55$                          | 不对称 Dice, 略微惩罚 FP       |
| Binary Focal loss | $\alpha = 0.25$, $\gamma = 2.0$                          | 聚焦难像素                   |
| Boundary loss     | Kervadec et al. 2019, lambda 在 0→0.1 over 50 epochs 线性升温 | 距离图监督边界                 |
| 总权重               | (0.5, 0.5, 0.1)                                          | tversky + focal + bd 加权 |

### 3.3 后续合入路径

1. 把 `WeightedTverskyFocalBoundary` 类从 `eppa-boundary` 分支 cherry-pick 进 `utils.py`
2. `train_model.py:184` 的 `criterion = WeightedDiceBCE(...)` 改为 `WeightedTverskyFocalBoundary(...)`
3. `Train_one_epoch.py` 增加 `tuple` 解包逻辑 (loss 返回 4-tuple `(total, l_tv, l_focal, l_bd)`)
4. 训练循环每 epoch 开头调用 `criterion.set_epoch(epoch)` 触发 lambda 升温

详见 git commit `0bc087b feat(loss): boundary-aware composite loss`.

---

## 4. 工程改造

### 4.1 训练断点恢复

**新增**: `Config.resume_path` (str, 默认 ''), `Config.resume_max_dice` (float, 默认 0.0).

**行为** (`train_model.py:208-237`):

- `resume_path` 非空时, 加载 state_dict, optimizer state, lr_scheduler state, `epoch`, `max_dice`, `best_epoch`,
  `epoch_history`
- 即使从 best_model resume, 仍创建新的 session 文件夹, **不覆盖原 best_model**, 保证溯源
- 旧格式 checkpoint (无 `lr_scheduler` 字段) 走 `lr_scheduler.step(start_epoch)` fast-forward
- 旧格式 checkpoint (无 `max_dice` 字段) 用 `Config.resume_max_dice` fallback

### 4.2 滚动 last_model + 完整 epoch_history

**新增** (`train_model.py:46-79, 278-294`):

```python
# 每个 epoch 都写一份 last_model-{model}.pth.tar (覆盖式), 不论 dice 是否提升
# best_model-{model}.pth.tar 仍只在 dice 突破时写
build_checkpoint_state(...)  # 将 epoch_history 序列化进 state dict
```

`epoch_history` 是 list of dict, 每 epoch 记录:

```python
{'epoch', 'train_loss', 'train_dice', 'train_iou',
 'val_loss', 'val_dice', 'val_iou', 'lr',
 'gate_stats': {up4: {mean, abs_mean, max, min}, ...}}
```

**bug 修复** (commit 中包含): 把 `epoch_history.append({...})` 移到 best/last save **之前**, 确保 best_model.pth.tar
包含自己当前 epoch 那一行 (旧逻辑会漏掉 best epoch 自己).

### 4.3 EPPA gate 诊断子表

**新增** (`train_model.py:82-100, 330-352`): `compute_eppa_gate_stats(model)` 在每个 epoch 末读取各 stage 的 `eppa.gate`
张量统计, 写入 `epoch_history`, 主表后追加 `EPPA Gate History` 子表打印.

**重要**: FreqEPPA 没有 `gate` 属性, 函数中已加 `if not hasattr(block.eppa, 'gate'): continue` 的防御检查 (
`train_model.py:95-99`), 所以 FreqEPPA 训练时该子表自动**不打印** (原表打印逻辑由
`if any(h.get('gate_stats') for h in epoch_history)` 守门).

### 4.4 Bark 推送 + 自动关机

**新增** (`train_model.py:22-30, 313, 332-334`):

- 训练开始: 推送 "🚀 训练开始"
- 每次刷新 best dice: 推送 "nb 兄弟" + 当前最高
- 训练结束: 推送 "✅ 训练结束"
- 训练完成后 `os.system("shutdown")` (云 GPU 节省费用)

### 4.5 thop 剥离

vanilla 训练脚本调用 `thop.profile(model, ...)` 计算 FLOPs. 但 thop 与 PEFT 包装的模块不兼容: 重复注册 hooks, 在
`.cuda()` 后留下指向 CPU 张量的 stale handles, 导致训练步抛 device 不匹配错. **解决**: 直接报告参数量, 不报 FLOPs (
`train_model.py:175-179`).

---

## 5. 超参数变更 (`Config.py`)

| 字段                               | vanilla LViT (`main`)                               | BetterLViT (`eppa`)                        | 变化原因                                            |
|----------------------------------|-----------------------------------------------------|--------------------------------------------|-------------------------------------------------|
| `epochs`                         | 1000                                                | 200                                        | 实测 200 内已收敛, 1000 浪费云 GPU 时间 (commit `31bc66e`) |
| `weight_decay`                   | (无)                                                 | `1e-4`                                     | 加 L2 正则降低过拟合 (commit `5742b5a`)                 |
| `text_max_len`                   | 硬编码 10 (在 `Train_one_epoch.py` 里 `text[:, :10, :]`) | 32, 暴露为 `Config.text_max_len`              | CXR-BERT 输出更长序列, 截短到 32 留出医学描述空间                |
| `text_encoder_name`              | (无, 静态查表)                                           | `microsoft/BiomedVLP-CXR-BERT-specialized` | 见创新 1                                           |
| `text_use_lora`, `text_lora_*`   | (无)                                                 | `True`, r=8, α=16, dropout=0.15            | 见创新 1                                           |
| `resume_path`, `resume_max_dice` | (无)                                                 | `''`, `0.0`                                | 见 4.1                                           |

---

## 6. 论文消融表模板

供毕业论文 Section "Ablation Study" 直接填:

| Model variant                         | Text encoder                     | Skip attention       | Loss                   | Val Dice ↑ | Val IoU ↑ |
|---------------------------------------|----------------------------------|----------------------|------------------------|------------|-----------|
| LViT (baseline, vanilla)              | `bert_embedding` (frozen lookup) | PLAM                 | Dice + BCE             | _          | _         |
| LViT + CXR-BERT (frozen)              | CXR-BERT (frozen)                | PLAM                 | Dice + BCE             | _          | _         |
| LViT + CXR-BERT + LoRA (q,v only)     | CXR-BERT + LoRA-qv (r=16)        | PLAM                 | Dice + BCE             | _          | _         |
| LViT + CXR-BERT + LoRA (all 6)        | CXR-BERT + LoRA-all (r=16)       | PLAM                 | Dice + BCE             | _          | _         |
| BetterLViT + EPPA-CBAM (failed)       | CXR-BERT + LoRA                  | EPPA-CBAM (gate=0)   | Dice + BCE             | 0.834      | 0.747     |
| BetterLViT + EPPA-CBAM + gate=0.1     | CXR-BERT + LoRA                  | EPPA-CBAM (gate=0.1) | Dice + BCE             | (待跑)       | (待跑)      |
| **BetterLViT + FreqEPPA**             | CXR-BERT + LoRA                  | **FreqEPPA**         | Dice + BCE             | (待跑)       | (待跑)      |
| BetterLViT + FreqEPPA + Boundary loss | CXR-BERT + LoRA                  | FreqEPPA             | Tversky+Focal+Boundary | (待跑)       | (待跑)      |

中间两行 EPPA-CBAM 是**有价值的负面消融**, 论文里应保留, 用于支撑 §2.2 的 "x_sa ≤ x 衰减约束" 论证.

---

## 7. 文件级 diff 概览 (`git diff main..HEAD --stat`)

```
Config.py          |  20 ++++-          (resume / lora / text_max_len)
Load_Dataset.py    |  83 ++++++++++--   (HuggingFace tokenizer 替换)
Train_one_epoch.py |  17 ++--           (text 改为 input_ids/attention_mask)
nets/BetterLViT.py |  70 +++++++++++++++ (新文件, 见创新 1)
nets/LViT.py       |  65 +++++++-----   (UpblockAttention 接 EPPA, text_seq_len 串联)
nets/Vit.py        |   6 +-             (CTBN3.in_channels 用 text_seq_len 参数)
nets/eppa.py       | 134 ++++++++++++++ (新文件, 见创新 2)
test_model.py      |  43 ++++++---      (适配 input_ids/attention_mask)
train_model.py     | 249 ++++++++++++++ (resume / history / Bark / 诊断)
utils.py           |  10 +--            (loss 仅 import 调整, 主体未变)
共 10 文件, +570 / -127 行
```

---

## 8. 不变量清单 (后续修改时务必保持)

写论文实施期间扩展模型, 这些约束**不要破坏**, 否则会静默错位 (CLAUDE.md 已罗列, 此处再列一遍便于检索):

1. **文本隐藏维度 = 768**. `LViT.text_module4 = Conv1d(768, 512, 3)`. 换文本编码器必须输出 768d.
2. **`text_max_len` 串联**:
   `Config.text_max_len → BetterLViT.__init__ → LViT.__init__ → VisionTransformer(text_seq_len=...) → Vit.CTBN3.in_channels`.
   不可单独改一个.
3. **Sigmoid 在模型内**: `LViT.last_activation = nn.Sigmoid()`. 所有 loss 在概率域上计算, 不要换 BCEWithLogits.
4. **EPPA 从零训练**. PLAM/EPPA-CBAM/FreqEPPA 的 state_dict 形状不兼容, resume_path 跨 EPPA 版本会失败.
5. **复合 loss 4-tuple 解包**: 如未来接入 `WeightedTverskyFocalBoundary`, `Train_one_epoch.py:71` 必须用
   `loss_output = criterion(...); loss = loss_output[0] if isinstance(loss_output, tuple) else loss_output`.
6. **`criterion.set_epoch(epoch)` 在每 epoch 开头**: 如果 loss 有 epoch-dependent 升温 (Boundary loss), 训练循环必须在每
   epoch 顶部调用, 否则 lambda 永远停留在 fallback 值.

---

## 9. 参考文献 (论文 BibTeX 候选)

```bibtex
@article{li2022lvit,
  title={LViT: Language meets Vision Transformer in Medical Image Segmentation},
  author={Li, Zihan and Li, Yunxiang and Li, Qingde and ...},
  journal={IEEE TMI},
  year={2023},
  note={arXiv:2206.14718}
}

@article{boecking2022making,
  title={Making the Most of Text Semantics to Improve Biomedical Vision-Language Processing},
  author={Boecking, Benedikt and Usuyama, Naoto and Bannur, Shruthi and ...},
  booktitle={ECCV},
  year={2022}
}

@article{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yelong and ...},
  journal={ICLR},
  year={2022}
}

@article{woo2018cbam,
  title={CBAM: Convolutional Block Attention Module},
  author={Woo, Sanghyun and Park, Jongchan and ...},
  booktitle={ECCV},
  year={2018}
}

@article{marr1980theory,
  title={Theory of Edge Detection},
  author={Marr, David and Hildreth, Ellen},
  journal={Proc. R. Soc. B},
  year={1980}
}

@article{peebles2023scalable,
  title={Scalable Diffusion Models with Transformers},
  author={Peebles, William and Xie, Saining},
  journal={ICCV},
  year={2023},
  note={DiT, source of zero-init last-layer trick}
}

@article{zhang2023controlnet,
  title={Adding Conditional Control to Text-to-Image Diffusion Models},
  author={Zhang, Lvmin and Rao, Anyi and Agrawala, Maneesh},
  journal={ICCV},
  year={2023},
  note={zero-init for safe-init residual}
}

@article{kervadec2019boundary,
  title={Boundary Loss for Highly Unbalanced Segmentation},
  author={Kervadec, Hoel and Bouchtiba, Jihene and ...},
  journal={MIDL},
  year={2019}
}

@article{salehi2017tversky,
  title={Tversky Loss Function for Image Segmentation Using 3D Fully Convolutional Deep Networks},
  author={Salehi, Seyed Sadegh Mohseni and Erdogmus, Deniz and Gholipour, Ali},
  journal={MLMI},
  year={2017}
}

@article{lin2017focal,
  title={Focal Loss for Dense Object Detection},
  author={Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and ...},
  journal={ICCV},
  year={2017}
}
```

---

## 10. 待办 / Open issues

- [ ] **接入 `WeightedTverskyFocalBoundary`**: 当前训练用的 `WeightedDiceBCE` 与 CLAUDE.md 描述的复合损失不一致. 这是
  §2.2 "饱和区监督信号弱" 的最直接补救.
- [ ] **跑 FreqEPPA 完整训练**: 验证 §2.4 的 identity-at-init 实证 (epoch 1 dice 不应低于 baseline 起点). 验证 sa > 1
  现象 (训练后某些边缘像素 sa 应 > 1).
- [ ] **新增 ca/sa 诊断**: 类似 EPPA gate 子表, 但记录 ca, sa 在验证集上的分布统计 (mean, max/min, 边缘 vs
  平滑区均值差).
- [ ] **(可选) Tversky α/β 翻向消融**: Covid19 ground-glass opacities 是漫散区域, FN 漏检代价高于 FP, 应考虑 α > β.
- [ ] **(可选) 深度监督**: decoder 中间层加辅助 segmentation head + 辅助 loss, 文献上常稳定 +0.5–1% Dice.

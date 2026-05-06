# EPPA — Edge-Preserving Pixel Attention 详解

> 文件位置: `nets/eppa.py`
> 用途: 在 LViT 解码器跳跃连接上，替换原版 `PixLevelModule (PLAM)`,
> 用更轻量、文本条件化、且对边缘友好的注意力门控 skip 特征。
> 当前分支: `eppa-boundary`,提交 `e58c67d feat(eppa): replace PLAM with Edge-Preserving Pixel Attention`。

---

## 1. EPPA 在网络里的位置

`nets/LViT.py` 里 4 个 `UpblockAttention` 各自持有一个 EPPA 实例:

```
                                 skip_x           (来自编码器对应层 + ViT 重建)
                                   │
                     ┌─────────────▼─────────────┐
   x ─► Upsample(2)  │   EPPA(skip_x, text)      │   ◄── text [B, L, 768]
        │            └─────────────┬─────────────┘
        │                          │ skip_x_att
        └──────── concat ──────────┤
                                   ▼
                        nConvs (3x3 conv ×2)  ─► 下一层
```

`UpblockAttention.__init__` 用的是 `in_channels // 2`,所以 4 个 EPPA 作用在如下通道数:

| stage | UpblockAttention `in_channels` | EPPA channels | reduction → `c_red` |
|-------|--------------------------------|---------------|---------------------|
| up4   | 1024                           | 512           | 64                  |
| up3   | 512                            | 256           | 32                  |
| up2   | 256                            | 128           | 16                  |
| up1   | 128                            | 64            | 8                   |

`text_dim` 全部传 `768`(`LViT.py:97 TEXT_DIM = 768`),即 CXR-BERT 的隐状态维度。EPPA 在此**未做文本通道压缩**,直接用 768 →
C 的 `Linear`。

> 注意: ViT 分支用的是经过 `text_module4..1` 级联压到 [512,256,128,64] 的 `text4..text1`;但 EPPA 接收的是*
*未压缩的原始 `text` (`[B, L, 768]`)**。两条路径的文本特征是不一样的。

---

## 2. 模块结构总览

EPPA 由 4 个子组件构成:

```
            x [B, C, H, W]                 text [B, L, 768] (可选)
              │                                  │
              ├──► GAP / GMP ──► shared MLP ──► (+) ◄── text[:, 0, :] · text_proj
              │                                  │
              │                                  ▼
              │                               σ(·)      ──► ca [B, C, 1, 1]
              │                                  │
              ├────────── ⊗ ─────────────────────┘
              │   x_ca [B, C, H, W]
              │           │
              │           ├──► avg/max over channel dim
              │           │
              │           ▼
              │       3x3 Conv ──► σ(·)  ──► sa [B, 1, H, W]
              │           │
              ├────────── ⊗ ──────────────► x_sa [B, C, H, W]
              │                                  │
              ▼                                  ▼
              x  ◄────────── + ◄── gate · x_sa  (LayerScale 残差)
                output
```

像 CBAM(Channel Attention + Spatial Attention 串联),但有 3 处关键改造:

1. **文本条件**注入到 channel logit;
2. spatial 用 **3×3** 而不是 CBAM 的 7×7,保留边缘梯度;
3. 输出走 **per-channel LayerScale 残差**,初始化为 0(训练起点是 identity)。

---

## 3. 逐部件解读

### 3.1 Channel Attention(通道注意力)

```python
self.ch_mlp = nn.Sequential(
    nn.Linear(in_channels, c_red, bias=False),
    nn.ReLU(inplace=True),
    nn.Linear(c_red, in_channels, bias=False),
)
```

经典 CBAM 范式:GAP 和 GMP 共享同一个 bottleneck MLP,结果**相加**(不是 concat)再过 sigmoid。

```python
avg_pool = x.mean(dim=(2, 3))                # [B, C]
max_pool = x.amax(dim=(2, 3))                # [B, C]
ch_logit = self.ch_mlp(avg_pool) + self.ch_mlp(max_pool)
```

- **为什么 avg 和 max 都要?** 平均池化反映"整体强度",最大池化反映"局部峰值"。医学影像上肺纹理是分布式的(用 avg)
  ,而病灶/边缘是稀疏强响应(用 max),两者互补。
- **为什么共享 MLP?** 减参,且让两条池化路径学到的"通道重要性"在同一空间里加和。
- **`reduction=8`**: 比 CBAM 默认的 `r=16` 更小,保留更多容量。`c_red = max(C // 8, 8)` 保证最浅层(C=64)还有 8 维
  bottleneck,不至于退化到 1-2 维。

### 3.2 文本条件融合(Text-conditioned channel logit)

```python
if text is not None and self.text_dim is not None:
    text_cls = text[:, 0, :]                 # [B, 768]
    ch_logit = ch_logit + self.text_proj(text_cls)
ca = torch.sigmoid(ch_logit)[:, :, None, None]
```

这是 EPPA 区别于纯 CBAM 的核心:**文本 logit 在 sigmoid 之前**与图像 logit 加和,整条通道路径只有 **1 次 sigmoid**。

**为什么不用 `sigmoid(image) * sigmoid(text)`?**
两个 sigmoid 输出都在 (0,1),期望约 0.5;乘起来期望 ≈ 0.25 → 会把输出系统性压缩到接近 0,残差信号几乎消失。EPPA 把两路 logit
**加在 sigmoid 之前**,等价于:

> ca = σ(image_logit + text_logit)

输出仍然完整覆盖 (0,1),且文本可以**正向**(放大某通道)或**负向**(抑制某通道),不会出现"双 sigmoid 折叠"。

**为什么用 `[CLS]` 而不是 mean-pooling?**

注释里写得很清楚(`eppa.py:25-27`):

> `[CLS] is used (not masked-mean) because CXR-BERT-specialized's [CLS]
>  is image-text contrastive aligned during pretraining and is by
>  construction free of [PAD] pollution.`

CXR-BERT-specialized 在预训练阶段做过图像-文本对比对齐,`[CLS]` 直接编码"整段报告"对应的图像语义;同时 `[CLS]` 不受
padding token 污染,而 mean-pool 会被 [PAD] 拉低除非显式 mask(成本和 bug 来源)。

> ⚠️ `CLAUDE.md` 把这条列为**不可破坏的不变量**:换 mean-pool 必须先理解 [PAD] 污染权衡。

### 3.3 Spatial Attention(空间注意力)

```python
self.sp_proj = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
...
sp_avg = x_ca.mean(dim=1, keepdim=True)      # [B, 1, H, W]
sp_max = x_ca.amax(dim=1, keepdim=True)      # [B, 1, H, W]
sa = torch.sigmoid(self.sp_proj(torch.cat([sp_avg, sp_max], dim=1)))
```

CBAM 原文用的是 **7×7** kernel, EPPA 用 **3×3**,这是 EPPA 名字里 "Edge-Preserving" 的核心理由。

- **3×3 = 经典边缘检测核大小** (Sobel/Laplacian/Prewitt 都是 3×3)。感受野 ±1 像素,刚好够检测"相邻像素强度跃变"。
- **7×7 会平滑边缘**: 大核相当于低通滤波,把局部梯度抹平。在 224×224 的胸片分割上,病灶边界往往只有几个像素宽,7×7 spatial
  attention 实测会"糊"。
- **输入 2 通道 = avg 和 max 在通道维上的融合**。avg 给"整体亮度地图",max 给"通道里最强响应位置",拼起来让 3×3 conv
  既看到背景也看到峰值,从而学到"这里像不像目标边界"。

> 注意此处 spatial 是**作用在 channel-attended 后的 `x_ca` 上**,不是直接作用在 `x` 上。等价于先选通道、再选像素,因子化为
`ca[c] * sa[h,w]`,从而以 GAP+3×3 conv 的低成本近似一个 per-pixel-per-channel 的注意力。

### 3.4 LayerScale 残差门控

```python
self.gate = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
...
return x + self.gate * x_sa
```

- **形状 `(1, C, 1, 1)`**: 每个通道一个独立标量门。
- **初始化为 0**: 训练第 0 步时 EPPA 输出 = `x + 0 = x`,即**完全 identity**。这是 LayerScale (CaiT, 2021) 的思路:
  让新模块在初始化时不破坏已有 backbone,然后让网络自己学要不要打开它(可正可负)。
- **没有非线性激活**: 梯度可以**直接**反传到 `gate`,前期不会被 sigmoid/relu 截断,新模块会"软启动"。
- **per-channel 而非全局**: 每个通道可以独立选择"用多少 EPPA 信号",更灵活。

> ⚠️ `eppa.py:33-36` 与 `CLAUDE.md` 都明确写了 **resume policy: train from scratch only**。
> 即使 init=0 让 EPPA 在初始化时是 identity,但**解码器后续模块(`nConvs`,以及深一层的 ViT/Up 块)是基于 PLAM-modulated
skip features 训出来的分布**,直接加载 PLAM checkpoint 会让它们一开始就接收到与训练分布不同的 skip 输入,触发"分布漂移"。所以
> EPPA 模型必须从零训练。

---

## 4. Forward 完整 shape walk

以 `up3` (EPPA channels=256, H=W=56)、batch=4、`text_max_len=32` 为例:

```
x        : [4, 256, 56, 56]      # skip_x 来自编码器
text     : [4, 32, 768]          # 来自 BetterLViT.encode_text

# Channel attention
avg_pool : [4, 256]              x.mean((2,3))
max_pool : [4, 256]              x.amax((2,3))
ch_logit : [4, 256]              ch_mlp(avg) + ch_mlp(max)
text_cls : [4, 768]              text[:, 0, :]
text_logit:[4, 256]              text_proj(text_cls)
ch_logit : [4, 256]              + text_logit
ca       : [4, 256, 1, 1]        sigmoid + 加两个 None 维
x_ca     : [4, 256, 56, 56]      x * ca   (broadcast)

# Spatial attention
sp_avg   : [4, 1, 56, 56]        x_ca.mean(dim=1, keepdim=True)
sp_max   : [4, 1, 56, 56]        x_ca.amax(dim=1, keepdim=True)
sp_in    : [4, 2, 56, 56]        concat
sa       : [4, 1, 56, 56]        sigmoid(3x3 conv)
x_sa     : [4, 256, 56, 56]      x_ca * sa  (broadcast on channel)

# Residual
gate     : [1, 256, 1, 1]        learnable, init=0
out      : [4, 256, 56, 56]      x + gate * x_sa
```

---

## 5. 与 PLAM(原 PixLevelModule) 对比

| 维度             | PLAM (`pixlevel.py`)                                                             | EPPA (`eppa.py`)                      |
|----------------|----------------------------------------------------------------------------------|---------------------------------------|
| 输入             | `x` only                                                                         | `x` + `text` (可选)                     |
| 通道注意力          | 1×1 conv on `x_avg_pool` 和 `x_max_pool`                                          | shared MLP on GAP+GMP                 |
| 文本条件           | ❌ 无                                                                              | ✅ [CLS] 投影,sigmoid 前融合                |
| 空间注意力          | 没有显式 spatial 分支,而是把 (avg, max, avg+max) 三标量过一个 `Linear(3→6→1)` 后再做逐像素 sigmoid 调制 | 显式 3×3 conv,边缘友好                      |
| 残差             | ❌ 直接 `y = x_output * x` (无残差,可能压缩信号)                                             | ✅ `x + gate * x_sa`,LayerScale init=0 |
| 双 sigmoid 风险   | ❌ 单 sigmoid                                                                      | ✅ 单 sigmoid(刻意设计)                     |
| 参数量(per stage) | 较小,但容量受限                                                                         | 略大(主要在 `text_proj`),但加入文本几乎免费         |

> PLAM 的主要问题: 通道注意力本质上只用了 3 维 bottleneck (avg/max/sum),容量过低,且没有残差,容易过度调制 skip 特征。EPPA
> 用更标准的 CBAM 容量做通道,留了显式 3×3 spatial 出来给边缘梯度,再用 LayerScale 把"加多少"交给学习。

---

## 6. 参数量估算

每个 EPPA(`reduction=8`,`text_dim=768`) 参数:

```
ch_mlp     : C * c_red * 2                       = 2 * C * c_red    (无 bias)
text_proj  : 768 * C + C                         = 768C + C         (有 bias,默认)
sp_proj    : 2 * 1 * 3 * 3                       = 18               (无 bias)
gate       : C                                                       
```

具体到 4 个 EPPA(C, c_red 见 §1):

| stage  | C   | c_red | ch_mlp | text_proj | sp_proj | gate | **小计**    |
|--------|-----|-------|--------|-----------|---------|------|-----------|
| up4    | 512 | 64    | 65,536 | 393,728   | 18      | 512  | ~459,794  |
| up3    | 256 | 32    | 16,384 | 196,864   | 18      | 256  | ~213,522  |
| up2    | 128 | 16    | 4,096  | 98,432    | 18      | 128  | ~102,674  |
| up1    | 64  | 8     | 1,024  | 49,216    | 18      | 64   | ~50,322   |
| **合计** |     |       |        |           |         |      | **~826K** |

绝大部分参数都在 `text_proj` (768→C),通道注意力本身极轻。

---

## 7. 关键设计决策清单(回顾)

1. **CBAM 风格 channel attention(GAP+GMP shared MLP)** — 工业级稳定,容量足。
2. **文本注入在 sigmoid 之前** — 避免双 sigmoid 折叠到 0.25,允许文本正/负调制。
3. **用 `[CLS]` 而非 mean-pool** — CXR-BERT-specialized 的 [CLS] 经过对比对齐,且天然 [PAD] 安全。
4. **3×3 spatial conv** — 与 Sobel/Laplacian 同尺度,保留 ±1 像素边缘梯度;不用 CBAM 的 7×7 因为会糊边。
5. **per-channel LayerScale,init=0** — 初始化时 identity,网络逐步自学打开;每通道独立、可正可负;无非线性,梯度直达。
6. **Train-from-scratch only** — init=0 不能阻止"PLAM 训练的解码器分布"和"EPPA 解码器分布"之间的不匹配。

---

## 8. 后续可探索方向(非当前实现)

- **跨尺度文本压缩**: 现在 4 个 EPPA 都吃 768-d 的 [CLS],浅层用了浪费;可在 `text_proj` 前共享一个 768→256 的下投影。
- **`reduction` 自适应**: 浅层 C=64 时 reduction=8 已经把 bottleneck 压到 8,边际容量太低;可以考虑 `min(C, 32)` 之类下界。
- **spatial attention 也注入文本**: 目前只有 channel 路径用了文本;若想做"文本告诉模型该看哪里",可以用
  `text_proj_spatial` 投到 `[B, 1, 1, 1]` 加到 `sp_proj` 输入上。
- **门控初始化非零的小值**: 若 train-from-scratch 时收敛太慢,可试 `gate.fill_(1e-4)`,以加速早期 EPPA 信号介入(代价:略微破坏
  identity 起点)。

---

## 9. 引用文件位置

- `nets/eppa.py:6-86` — EPPA 模块定义
- `nets/LViT.py:61-74` — `UpblockAttention` 调用 EPPA
- `nets/LViT.py:97-101` — 4 个 `UpblockAttention` 实例化(`TEXT_DIM=768`)
- `nets/LViT.py:141-144` — forward 中传入未压缩的 `text` 给 4 个 up 块
- `nets/pixlevel.py` — 被替换的原版 PLAM,留作 ablation 参考
- `CLAUDE.md` — Innovation #2 EPPA 项 / "Train from scratch" 不变量声明
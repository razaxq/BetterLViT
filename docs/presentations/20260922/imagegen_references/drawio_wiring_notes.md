# FSDR 与 RACE：draw.io 重画说明

这三张图由内置 ImageGen 生成，供论文式配色、立体特征块和布局参考。
图片没有替换 PPT，也不是实验预测结果。最终矢量图的连接应按下文核对。

## 总览图需手动修正、补足的位置

1. CNN 第四层仍被生成器误写为 C3，应改为 **C4**。五层依次为 C1、C2、C3、C4、C5。
2. ViT 总览省略了同尺度横向信息连接。重画时补上各层 Down ViT 到对应 Up ViT 的横向连接，并按实现表达 CNN 特征对 Down ViT 的引导。
3. `V_i` 与 `T` 使用同名端口表示输入；它们不是相邻解码器之间的纵向传递信号。纵向路径传递解码特征。
4. 绿色 up 模块代表整个解码阶段，不仅是 FSDR。需要进一步展开时，按 `Upsample → FSDR → Concat → Conv ×2` 绘制。
5. 图中的 Up ViT 1–4 按空间尺度编号，不是执行时间顺序；执行顺序由深到浅。

## 与代码一致的总览连接

- 图像 → CNN C1 → C2 → C3 → C4 → C5。
- 每个 CNN 跳跃：`C_i → RACE_i → C'_i → FSDR_i`。RACE 不改变继续向下的编码器路径，也不将 C'_i 反馈给 ViT。
- ViT 重建特征 `V_i` 和文本 T 分别进入对应 FSDR；不能先与 C'_i 相加。
- 解码顺序：C5 进入 up4，随后 up3、up2、up1，最后输出分割。
- 每个 up 阶段先上采样得到 D，再由 `FSDR(C'_i,V_i,D,T)` 得到 Y、D′，拼接 Y 与 D′ 后两层卷积。
- 报告经冻结 CXR-BERT 得到 T。T 为 ViT、RACE 和 FSDR 提供条件。

## FSDR 图

- C* 表示 FSDR 的 CNN 输入：组合模型使用 C′，仅 FSDR 对照使用 C。
- C* 在 Haar 分解之前分出原始恒等旁路，直接进入最终加法。
- Haar 给出 C_low 和 C_high；语义分支接收 C_low、V_low、D′_low、T。
- 语义分支分别输出 R_sem 与 Support S；R_sem 直接加到 Y，S 用于细节精炼。
- C_high 与 S 进入细节分支，其 αd R_detail 直接加到 Y。
- `Y = C* + R_sem + αd R_detail + R_adaptive`。
- `R_sem = αp V_low + R_channel + αr R_region`。
- up4/up3 的低频上下文预测空间滤波混合权重。**滤波器实际作用于原始 C* 和 D**，重画详细版本时须补这两个输入，不能只画权重直接生成特征。
- `D′ = D + α[LP_low(D)−D]`；`R_adaptive = β[C*−LP_high(C*)]`。
- D′ 经 Haar 得到 D′_low，供语义分支使用；D′ 本身供解码器拼接。D′ 不加入 Y。
- up2/up1：D′=D，R_adaptive=0；基础 Haar 语义/细节分支仍启用。

## RACE 图

- T → 掩码均值 → 槽位头 → 六区域预测 z。
- z 与独立的六区解剖基底 B 共同形成空间先验 P。
- C → 视觉证据 E；E 按 B 池化后与 z 比较，得到一致性 A。
- P、E、A 各自进入门控 `G=P×E×A`。
- C → 可学习卷积残差 R(C)，不是 C−E。DWConv/PWConv 后还有归一化与 GELU，当前图省略了这些常规操作。
- `C′ = C + s G ⊙ R(C)`；原 C 恒等旁路只进入最终加法。
- `s=0.15 tanh(a)`，a 零初始化。图中省略计数辅助头；完整模块训练还包含计数、视觉与一致性等辅助目标。
- 解析器目标用于辅助训练，不作为推理输入。

## 实现来源

图形布局参考用户提供的 LViT 论文图。结构核对基于本项目的
`nets/LViT.py`、`nets/eppa.py`、`nets/race_fuse.py`，代码整合版本
`b71218b52240c60b2e47c25d584448f0dde2f73d`。
生成提示词见 `generation_prompts.json`。图像生成存在随机性，不能通过提示词保证逐像素复现。

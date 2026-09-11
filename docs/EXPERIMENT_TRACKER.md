# BetterLViT 实验台账

更新时间：2026-09-11（Australia/Sydney）

本文件是后续实验配置、Git 溯源、验证结果和推进状态的唯一人工维护台账。停止继续维护 `改动计划.xlsx`；旧工作簿仅作为历史快照保留。

## 记录规则

- 论文主口径为逐样本 macro Dice/IoU；不在本台账维护 FMISeg 论文的 micro 口径。
- 正式实验必须使用独立完整 Git SHA 和实验标签，运行环境、检查点 `source_git_commit` 与结果 JSON 必须一致。
- 机制筛选 pilot 只使用 validation：`AUTO_EVALUATE=0`、`TEST_SPLIT_ALLOWED=0`。未通过阶段门不得扩展或访问 Test。
- 当前及后续架构实验不使用 LoRA。Focal 可以使用；禁止使用 boundary loss，正式配置必须保持 `boundary_loss=0.0`。
- C0/P5 是已完成的 Dice/Tversky 配对验证：`0.5 * Dice + 0.5 * Tversky`，Tversky 的 FP/FN 权重为 `0.7/0.3`。这不代表后续主线禁用 Focal。

## 文字引导方向：研究与已有方法核查完成（2026-09-11）

用户要求第二创新点建立在文字之上，并继续核查可行性与已有研究。完成[研究报告](../repro_archive/20260911/text_grounding_research/REPORT.md)：覆盖C2Seg、SSA、RecLMIS、ARSeg、CausalCLIPSeg、FairVLM、TGC-Net、InstAlign、CROSS及体积保持分割等近邻，明确拆短语、双向注意力、空间反事实或保量层本身均不是新点。

本地Train静态审计5716条：273种原文、240种分词序列，含特殊token最长24，**无32-token截断**；严格解析1256条非对称双侧、1155条单侧，36组相同词汇但不同位置绑定覆盖1236条。语法成功率不是医学准确率；本次未读取mask内容或新增Val/Test推理，数据/源码指纹已记录。

候选收窄为“文本关系引导的概率重分配”：同图像下真实/关系替代描述产生局部差分，以原输出软前景总量约束新增修正。NumPy验证总量、identity、常数抵消及隐式梯度；也给出错误证据仍可退化、软总量不保证二值面积的反例。**仅为组合式研究候选，尚无已成立新颖性或IoU增益，无新训练启动。** 建议先做主文字通路干预，再进行普通匹配/差分×有无投影的受控筛选，最后匹配种子和冻结Test。RS3继续原注册流程及22:00末检，不改变现有实验门或预约。

## RS1/RS2完成，RS3按冻结顺序接续（2026-09-11）

**RS3首次预约检查完成：** 悉尼2026-09-11 17:13:13.659，一次短连接确认完整4/80轮，第5轮至少340/357训练批次，GPU100%、显存17402MiB，来源SHA/manifest一致且跟踪文件干净，日志尾部无异常。第2—4轮平均217.763331秒，加120秒后续诊断余量，预测今天21:47:50训练结束。同一heartbeat `betterlvit` 已改约**22:00唯一末检**并核验实际配置，检查1/2；之后不再中途连接。没有完整RS3结果，未访问Test。[首次检查与预测](../repro_archive/20260911/regional_supervision_execution/rs3_first_snapshot.json)。

**RS3已提交：** 悉尼16:55:06.973提交80轮后台训练，PID261154，来源`f79842331e4e5e41526ed66da31ec174ea4a61b2`，tag `experiment-rs3-regional-80e-seed1219-20260911`。固定λ=0.128312，局部阳性/空背景分组归一化；启动前训练盘余4,284,715,008字节，fs18,559,782,256字节。同一heartbeat首检已约**今天17:11悉尼时间**并核验配置。当前只有提交回执，检查0/2，无完整RS3结果；首轮完成后再作阶段判断。归档流程已包含RS3−RS2配对比较，用于按原计划评估分组独立收益，不改变训练或门槛。[RS3启动](../repro_archive/20260911/regional_supervision_execution/rs3_launch.json)。

**RS2完成并独立核验：** 悉尼16:36:27.368完成80轮，16:38:13.553完成1429张Val导出，Best75，来源`2a389922cadb31a8bf7660bc33b3cb1ae94f20ab`。唯一末检16:46:13.188距训练结束9.76分钟，符合半小时要求；检查2/2，训练/Val退出码0，无Test访问。80轮学习率、来源、逐图均值、四次Train诊断状态恢复及Best/Last均核验。

| 模型 | Val IoU | Val Dice | Precision | Recall | Brier |
|---|---:|---:|---:|---:|---:|
| R2 | 72.6654% | 82.4034% | 80.5086% | 88.0502% | 0.021162 |
| RS1整图软IoU | 73.0764% | 82.7039% | 80.1376% | 89.2672% | 0.021052 |
| RS2局部软IoU | 72.8395% | 82.5258% | 80.8231% | 88.0082% | 0.020535 |

RS2−R2 IoU +0.1741个百分点，95%图像配对区间[-0.1181,+0.4575]；最低GT面积四分位Dice−0.2540、Recall−0.4992个百分点。R2性能门3/7通过，整体失败。RS2−RS1 IoU−0.2369个百分点，区间[-0.5454,+0.0522]，也未过区域增量门。不能称稳定增益，不能将点估计差解释为已证明RS2劣于RS1。[完整RS2报告](../repro_archive/20260911/regional_supervision_execution/rs2_results/REPORT.md)。RS3按原方案完整执行，不根据前两组修改系数或窗口。

RS2已备份至HF `2a389922/`，20文件、1,694,460,839字节大小/Xet双端核验通过，11项本地下载run产物亦匹配。为RS3腾空间，将RS1已备份Last（845,347,135字节）迁至`/root/regional_model_archive/b4dd566a/`，原路径符号链接可访问、Best原位保留；活跃引用审计、Xet/SHA256及检查点加载核验通过。训练盘可用3,439,378,432→4,284,727,296字节，系统盘3,974,262,784→3,128,893,440字节，fs维持18,559,782,256字节；不删除模型。[迁移回执](../repro_archive/20260911/regional_supervision_execution/storage_migration_rs1_last.json)。

**RS2首次预约检查完成：** 悉尼2026-09-11 11:59:42.365，一次短连接确认完整4/80轮，第5轮已进入Val，GPU83%、显存17402MiB，来源SHA/manifest一致且跟踪文件干净，日志尾部无异常。第2—4轮平均216.155973秒，加120秒后续诊断余量，预测今天16:32:12训练结束。同一heartbeat `betterlvit` 已改约**16:45唯一末检**并核验实际配置，检查1/2；之后不再中途连接。尚无完整RS2结果，未访问Test。[首次检查与预测](../repro_archive/20260911/regional_supervision_execution/rs2_first_snapshot.json)。

**RS2已提交：** 悉尼11:41:35.575提交80轮后台训练，PID249881，来源`2a389922cadb31a8bf7660bc33b3cb1ae94f20ab`，tag `experiment-rs2-regional-80e-seed1219-20260911`。56窗口/28步长局部软IoU自然平均，λ=0.128312；提交前训练盘余5,155,667,968字节、fs18,559,782,256字节，满足空间门。同一heartbeat首检改约**今天11:57悉尼时间**并核验配置，当前只有提交回执、检查0/2，无RS2完整结果。RS3等待前组归档后接续；RS1检查2/2已用完，不重复检查。[RS2启动](../repro_archive/20260911/regional_supervision_execution/rs2_launch.json)。

RS1于悉尼11:11:41.771完成80轮、11:13:25.457完成1429张Val导出，Best69，来源`b4dd566ae472079c55e41cffd7727060bcfd6bf5`。唯一末检11:28:34.230距训练结束16.87分钟，符合半小时要求；检查2/2，无Test访问。逐图指标、80轮学习率、来源及四次Train诊断状态恢复均核验通过。[RS1完整报告](../repro_archive/20260911/regional_supervision_execution/rs1_results/REPORT.md)。

| 模型 | Val IoU | Val Dice | Precision | Recall | Brier |
|---|---:|---:|---:|---:|---:|
| R2 | 72.6654% | 82.4034% | 80.5086% | 88.0502% | 0.021162 |
| RS1整图软IoU | 73.0764% | 82.7039% | 80.1376% | 89.2672% | 0.021052 |

IoU +0.4111个百分点，图像配对95%区间[+0.1181,+0.7023]；Dice +0.3004。Precision −0.3710个百分点，因此只通过6/7项筛选门，不能称全面过门或稳定增益。按预先计划继续RS2/RS3，保持共同λ=0.128312、80轮和原配方；不因本结果修改后组。RS1已完整备份至HF Bucket `b4dd566a/`，21文件、1,694,955,060字节的路径/大小/Xet哈希双端核验通过，11项本地下载产物亦匹配。原Best/Last保留，fs18,559,782,256字节、训练盘余5,155,680,256字节；后组实际提交证据以执行入口为准。

继承的训练IoU用>=0.5/float32，导出用>0.5/float64；一次完成模型的Val-only诊断发现1个背景像素恰为0.5，完整解释两者1.135662367e-7差值。原训练IoU与逐图导出复算均精确相同，检查点字节哈希不变，没有修改原始指标、阈值或冻结源码。

### 先前启动和首检记录

**RS1首次预约检查完成：** 悉尼2026-09-11 06:35:28.102，一次短连接确认完成4/80轮、正在第5轮，GPU92%、显存17402MiB，来源SHA/manifest一致且跟踪文件干净，无日志尾部异常。第2—4轮平均220.411700秒，加120秒后续诊断余量，预测今天11:14:25训练结束。同一heartbeat `betterlvit` 已改约**11:27唯一末检**，app实际配置核验通过，检查1/2，之后不再中途连接。无完整实验成绩或Test访问。[首检与预测](../repro_archive/20260911/regional_supervision_execution/rs1_first_snapshot.json)。

**RS1已提交、首检已预约：** 悉尼2026-09-11 06:18:07.393提交80轮后台训练，PID238273，SHA `b4dd566ae472079c55e41cffd7727060bcfd6bf5`，tag `experiment-rs1-regional-80e-seed1219-20260911`。当前仅提交回执，尚未首检确认健康，无完整结果或Test访问，检查预算0/2。当前任务heartbeat `betterlvit` 已实际创建，首检约06:34，app配置与目标thread核验通过；届时据实测轮时改约唯一末检。[启动与预约](../repro_archive/20260911/regional_supervision_execution/rs1_launch.json)。

全部预检通过：11项loss行为、5项阶段门；历史R2五步精确复现，三组初始化/输入一致，各组关闭loss后的正式分组优化器五步与R2逐值一致，插入Train诊断前后亦逐值一致。峰值分配约15.37GiB。RS2 `2a389922cadb31a8bf7660bc33b3cb1ae94f20ab`、RS3 `f79842331e4e5e41526ed66da31ec174ea4a61b2`已冻结部署并推送，尚未提交；按固定顺序归档前组后接续。启动前训练盘余6,872,870,912字节，共享fs18,559,782,256字节，模型保留。

用户明确要求“启动训练”。已从冻结R2派生三项最终输出监督：RS1整图软IoU、RS2局部软IoU（56窗口/28步长）、RS3阳性/空背景分组。三组各80轮、seed1219、原R2单次余弦与Dice/Focal，不新增网络头；按顺序完整执行首轮，再按计划决定后续。[执行入口、校准与证据](../repro_archive/20260911/regional_supervision_execution/README.md)。

固定Train-only校准得到共同λ=0.128312，初始12项批次×模式最大新增/主logit梯度比0.0999999；这是输出梯度尺度，不是共享参数梯度或IoU证据。源码、独立manifest和实验tag已推送GitHub并核验；最终CUDA预检及正式启动状态以下续记为准。

## 区域监督实验顺序已制定（2026-09-11）

用户同意方向并要求先给出实验顺序。已记录[阶段计划](REGIONAL_SUPERVISION_EXPERIMENT_PLAN_20260911.md)：预检及R2兼容性验证 → RS1整图软IoU → RS2局部软IoU → RS3局部阳性/背景分组；首轮固定3次80轮、seed1219，原R2单次余弦及Dice/Focal。将分组对照纳入首轮的依据是已知Train背景窗口比例，三组不依据中间数值临时改变。

按既定R2性能门筛选，局部设计还需相对RS1正向增量，之后才进入ARS/ITSRS受控适配、必要消融及2027/3407匹配种子，最终统一Test。当前只是计划，loss系数与运行manifest待工程预检后冻结；RS1—RS8均未启动、无新增训练或评估结果，未设置训练定时。

## 区域监督研究与Train形态审计（2026-09-11）

完成区域监督原始文献研究、合成反例和5716张Train mask的CPU只读审计。建议优先检验直接施加于最终分割图的局部重叠监督；S2的存在/占比目标可能忽略区域内等面积错位，不能代表全部区域监督方法。全图Dice已存在于R2，必须用“附加整图软IoU”对照区分目标对齐与局部粒度的作用。[完整研究报告与来源](../repro_archive/20260911/regional_supervision_research/REPORT.md)。

8连通统计共10781个区域，每图中位数2，只有4个区域小于16像素，不支持直接假定数据主要存在大量孤立微小病灶。固定非重叠28/56/112窗口的空背景比例分别69.76%/48.79%/18.05%；因此固定窗口是优先候选，但须明确空窗口惩罚与图内归一化。统计使用原224预处理、无增强，不能当成临床病灶实例或拟议重叠窗口的训练分布。[原始审计](../repro_archive/20260911/regional_supervision_research/train_region_audit.json)。

文献基线包括TPAMI 2023 ARS、2026 ITSRS，以及局部Dice、连通实例与区域对比研究；局部重叠和自适应FP/FN权重已有先例，不能直接称原创。候选顺序为R2＋整图软IoU、R2＋56窗口/28步长局部软IoU，再按结果检验前景/背景分组归一化及ARS/ITSRS适配。保留原80轮单次余弦和主Dice/Focal，无LoRA、boundary loss或新增网络头。当前是研究建议，系数及正式运行协议尚未冻结，未接入模型或启动训练，也未新增Val/Test推理。

审计源码提交`e862fbd655012e3f07f1a8b209f97ec0219a4620`，原模型来源保持S2冻结SHA；执行源码哈希与远端返回值一致。合成例子验证同样存在/占比可以对应IoU 1或0，示例仅说明压缩目标的局限，不能据此称S2整体没有空间监督。后续候选仍需原Val筛选及匹配种子复验，最终以Test macro IoU报告，尚无新增益结论。

## R2/S1/S2：按用户要求补充Test（2026-09-11）

用户明确要求三组测试集结果，故本次在保留原Val筛选失败事实的前提下新增S1/S2 Test；未训练新模型、未重选Best或调整阈值。统一seed1219、80轮单次余弦、阈值0.5、2113张Test、逐图macro。R2复用同种子历史Test，S1 Best67、S2 Best75，数据加载及推理/指标计算与R2原评估器AST一致。[完整报告](../repro_archive/20260911/visual_aux_test/REPORT.md)。

| 配置 | Test IoU | Test Dice | Precision | Recall |
|---|---:|---:|---:|---:|
| R2 | 75.8098% | 84.4236% | 82.6084% | 89.1621% |
| S1 | 75.8009% | 84.4069% | 82.5748% | 89.2971% |
| S2 | 75.9991% | 84.5226% | 83.3169% | 88.5354% |

S1−R2 IoU−0.0089个百分点；S2−R2 +0.1893，10000次图像配对bootstrap95%区间[-0.0492,+0.4281]跨0，Dice+0.0990。S2−S1 IoU+0.1982，区间[-0.0630,+0.4537]跨0。S2提高precision但降低recall，不能凭本次单种子Test宣称稳定增益或第二创新成立。

两组退出码0、完整JSON/日志SHA验证通过，三组文件名及GT面积逐图匹配，四项指标从整数交并计数复算通过。评估代码SHA `18757a28793206573f87239f537245e22b7e8ac2`，tag `evaluation-r2-s1-s2-test-seed1219-20260911`；S1/S2训练来源保持原SHA。按预测仅一次短连接收集，距评估完成103.06秒，未更改历史训练检查次数，未恢复训练定时。S2-T仍只有原型，尚无训练或测试结果。

## S2区域目标诊断与修复原型（2026-09-11）

完成全部5716张Train mask的CPU只读审计。现有“先池化mask、再乘缩放区域图”使28×28尺度877个派生存在标签与原分辨率区域不一致，涉及808张图像；其中482个阳性区域变为阴性。一次固定legacy几何增强抽样仍有835个不一致标签。这里是跨尺度派生目标不一致，不是原始标注错误，也尚未证明它造成多少IoU损失。[完整证据与修复方案](../repro_archive/20260911/s2_repair_audit/README.md)。

已实现独立原型：由增强后的原分辨率mask及同步区域图一次性生成presence/area/valid，供四尺度共享。4项CPU回归检查通过，源码哈希保存在检查结果中。原型尚未接入正式模型，未启动新训练、未加载权重、未访问Val/Test图像。建议第一候选S2-T只修目标，保留S2原聚合、系数和80轮配方；仍须通过原R2筛选门，不能把工程修复当作已成立的创新或增益。历史S1/S2失败结论保持不变。

现有Train梯度快照复算：presence的64个批次×尺度观察中20个与主loss负余弦，occupancy为0，辅助合计仅1个；这不支持“辅助梯度整体过大”。下一步若修正目标仍无效，再隔离presence/occupancy，最后考虑局部证据汇聚。没有恢复训练定时。

## R2视觉辅助监督消融：最终筛选结论（2026-09-11）

**完整归档与定时结束：** S2的20文件、1,698,369,075逻辑字节已添加HF的`7defc637/`并逐文件验证大小/Xet哈希，服务器与独立本地列表一致；连同S1共39文件、3,396,417,612逻辑字节。原Best/Last均保留，训练盘余6,881,239,040字节，共享fs18,559,782,256字节。两组检查均2/2且末检距真实训练结束均不足30分钟；未新增训练或Test，heartbeat `betterlvit` 已确认删除、配置文件不存在。[完成及云端核验证据](../repro_archive/20260910/visual_aux_execution/finalization.json)。

**两组均未通过，不扩展：** S2完成80轮，Best75，1429张Val macro IoU **72.7415%**、Dice **82.4575%**。对R2分别+0.0761/+0.0541个百分点，IoU95%图像配对区间[-0.2129,+0.3642]跨0；相对S1 IoU−0.1588个百分点，区间[-0.4633,+0.1433]跨0。R2门与区域增量门均失败；S1此前亦未过门，不进入2027/3407、多种子Test或150轮，不修改冻结系数及阈值。尚未确立第二创新成果。[阶段总结与复算](../repro_archive/20260910/visual_aux_execution/SUMMARY.md)。

S2对S1 precision+0.7128、recall−0.9725个百分点；每图FP减少62.34像素、FN增加33.12像素。最小GT总面积组358张的IoU/Dice/recall分别−1.2627/−1.0698/−2.7515个百分点，当前区域组合表现为减少误检但增加漏检。全部4287条模型×图像计数由导出重建，四项指标复算误差0；四次Train诊断有实际视觉梯度但不能当作IoU因果分解。存在/占比共同加入，尚不能区分两者贡献。

S2训练03:36:20.433、Val03:38:03.583完成，03:48:11.268末检（悉尼），距训练结束710.834秒，预算2/2；检查时GPU空闲，来源与冻结SHA一致、跟踪文件干净。80轮LR/Best选择/逐图均值/四次诊断状态均核验通过，S2完整HF备份已核验完成。当前没有新Test结果。[S2证据](../repro_archive/20260910/visual_aux_execution/s2_results/REPORT.md)。

## R2视觉辅助监督消融历史执行记录（2026-09-10）

**S2首次预约检查完成：** 悉尼2026-09-10 22:47:25.853，一次短连接确认已完成4/80轮、正在第5轮，GPU88%、显存17,840MiB，源码SHA一致且跟踪文件干净；三项辅助损失正常记录。第2–4轮平均225.504738秒，加四次Train诊断预留120秒，预测9月11日03:33:50训练结束。同一heartbeat已改约**9月11日03:46**唯一末检，app实际配置核验通过，检查1/2，之后不再中途连接。尚无完整实验结果，无Test访问。[首检与预测](../repro_archive/20260910/visual_aux_execution/s2_first_snapshot.json)。

**S2已提交：** S1归档与HF核验推送后，2026-09-10 22:30:42.609（悉尼）提交S2的80轮训练，PID219448，SHA `7defc637e36974fabd0f47763275c90f617e5f53`、tag `experiment-s2-r2-regional-80e-seed1219-20260910`；仅提交回执，检查预算0/2，尚未确认运行健康。保持原定三项监督系数，无Test访问。同一heartbeat已改约今天22:46首检，app配置核验通过；届时依据S2实测轮时预约唯一末检。[执行记录](../repro_archive/20260910/visual_aux_execution/README.md)。

**S1云端归档已核验：** 19文件、1,698,048,537逻辑字节已添加HF Bucket `razaxq/BetterLViT` 的 `1f7edb78/`，包含Best/Last、完整冻结源码、manifest、日志/Val/历史/诊断及对照分析，所有文件大小/Xet哈希在服务器与独立本地列表双重核验；11份本地下载原始文件哈希亦与服务器一致。原模型保留，训练盘可用8,602,071,040字节，共享fs18,559,782,256字节。[云端核验](../repro_archive/20260910/visual_aux_execution/s1_results/hf_upload_verified.json)。

**S1完成、筛选未通过：** 80轮，Best67，1429张Val macro IoU **72.9003%**、Dice **82.6077%**，对同seed R2分别+0.2350/+0.2043个百分点。IoU95%图像配对区间[-0.0330,+0.4986]个百分点跨0，未达+0.3门槛；其他5项非退化条件通过。不能称稳定提升，不扩展S1多种子或Test；按预注册仍继续S2。训练22:13:02.833结束，Val22:14:48.068完成，22:19:31.670末检（悉尼），距训练结束388.837秒，检查2/2；80轮LR、Best选择、来源SHA、逐图均值与四次Train诊断不干扰状态均核验通过。HF备份已核验完成。[完整报告](../repro_archive/20260910/visual_aux_execution/s1_results/REPORT.md)。

用户同意执行C8机制分析提出的对照。已冻结S1（R2＋像素BCE，有效系数0.02）和S2（再加区域存在BCE 0.01、区域占比MSE 0.005）；各80轮单次余弦、seed1219，原Dice/Focal、增强和优化器分组。先S1完成归档再S2，不根据S1数值改变第二组配置。训练成功自动导出1429张Val，本阶段无Test；最终性能仍以筛选及匹配种子复验后的统一Test为准。[冻结计划与执行入口](../repro_archive/20260910/visual_aux_execution/README.md)。

启动准备全部通过：7项监督行为检查、6项判定门检查；新R2与历史R2初始化/输入/五步预检逐值一致；S1/S2共享主干、首输出和随机状态与R2一致，共同辅助头初始化一致。各组插入32张Train诊断前后的五步输出/loss完全相同；S2关闭辅助loss后与正式优化器R2五步完全相同。这些是工程检查，尚非增益结果。四次Train遥测注册在20/40/60/80轮，保留模型、梯度及随机状态。

- S1源码 `1f7edb7858345b6b3ef0736291bb5cf92a2f712f`，tag `experiment-s1-r2-pixel-80e-seed1219-20260910`。
- S2源码 `7defc637e36974fabd0f47763275c90f617e5f53`，tag `experiment-s2-r2-regional-80e-seed1219-20260910`。
- 两组来源、标签均已推送GitHub，远端跟踪文件干净。

**S1已提交：** 悉尼2026-09-10 17:09:02.700后台提交，PID208088，预算0/2；尚未首检确认健康，无新指标。S2待S1完成归档后接续。当前任务heartbeat `betterlvit` 已实际创建为ACTIVE，首检预约17:25，与app保存配置及目标任务核验一致；首检后按实测轮时更新同一预约至预测结束后约12分钟。[提交及预约记录](../repro_archive/20260910/visual_aux_execution/README.md)。

**S1首次预约检查完成：** 2026-09-10 17:25:51.738（悉尼）自动触发的一次短连接确认已完成4/80轮、正在第5轮，GPU93%、显存17,840MiB，来源SHA一致且跟踪文件干净。第2–4轮平均219.600946秒，加四次Train诊断预留120秒，预计今天22:04:23结束；同一heartbeat已改约22:17唯一末检，app实际配置核验通过，预算1/2。无新增Test访问或完整实验成绩，S2待前组归档接续。真实结束至检查的30分钟约束将在末检核对。[首检与预测证据](../repro_archive/20260910/visual_aux_execution/s1_first_snapshot.json)。

候选对R2须Val IoU≥+0.003且95%配对区间下界>0，同时总体Dice/precision、最小GT总面积组Dice/recall不降、Brier不升。S2的区域增量另须相对S1 IoU≥+0.001及相同非退化条件。通过者才复验2027/3407；普通深监督收益不直接称第二创新。[完整预检证据](../repro_archive/20260910/visual_aux_execution/preflight_verified.json)。

清理前审计8个派生视觉探针缓存及活跃进程/打开文件，仅删除可重建NPY共6,336,072,704逻辑字节；训练盘可用从3,990,876,160增至10,326,982,656字节。模型、原数据与结果保留，共享fs清理前后均18,559,782,256字节，低于20GB实际字节限制。每组最多两次预约短连接，首检预测末检，不持续SSH。

## C8提升原因分析（2026-09-10）

已完成历史2113张Test/1429张Val逐图复算、C4/C8/P10完整80轮历史提取、冻结源码检查及固定32张Train的无更新梯度诊断。[完整分析报告、代码和图表](../repro_archive/20260910/c8_mechanism/REPORT.md)。本轮没有重训、新增Test推理或改动模型权重。

C8对C4 Test macro IoU为75.8612%对75.6396%（+0.2216个百分点），Dice为84.4313%对84.2989%；IoU原始图像区间[−0.0004467,+0.4463012]个百分点跨0，按474个文件名subject ID聚类抽样也跨0。误检每图减少29.39、漏检增加14.74；逐图对称代数分解为FP变化+0.295425、TP/FN变化−0.073875个百分点，合计实际增益。原本预测面积不足的图像平均漏检反而减少，不能解释成所有图像统一收缩。分层为事后描述，不能当成新确认性检验。

实际视觉梯度来自像素BCE/区域存在BCE/占比MSE，有效系数0.02/0.01/0.005。Best检查点上四批平均特征梯度范数相对主loss分别约0.91–1.02%/0.51–0.79%/0.023–0.032%；新增文本提及/数量项无视觉梯度连接，独立槽头有非零梯度。关闭辅助分支后32张Train预测逐值一致，完整模型状态哈希前后相同。局部eval梯度快照不能代替训练全过程消融或解释各loss的IoU因果贡献。

C4/C8逐轮LR完全相同、Best均80、Test阈值均0.5，评估脚本只差模型白名单；但C8最后10轮Val IoU均值70.9111%略低于C4的70.9420%，不能宣称整个训练后期更稳定。最合理机制候选是多尺度视觉监督的表示约束；像素/存在项值得优先拆解，占比项必要性尚不清楚，文本辅助和推理路由不能获得这项增益归因。

成功诊断源码 `3c3d53fb91dd74c8a9481ab65caef35117e04154`，原C8训练源码仍为 `21606e02c2d64ae950c0f243f55163f58cbedf83`。下一步建议R2原模型、R2+像素监督、R2+三项视觉监督的同配方对照，必要时再拆存在/占比；本轮没有启动。这是待验证研究方向，尚非第二创新点成果。

## 第二创新点：局部重编码可行性对照（2026-09-10）

用户要求解决第二创新点。重新审查后，旧线性分歧探针能力弱、修正幅度受限，不能据此否决所有局部编码。已实现两组同容量、同初始化、同优化预算的冻结R2分支：B重新解读现有decoder/CNN/语义特征及原概率图，C仅将输入编码器的概率图换成原图。每组192,129参数，原Dice/Focal，固定不确定性区域40/196块，末层零初始化但不限制logit残差幅度；不使用LoRA、边界loss或新监督。普通重编码不是已确立的创新。[预注册计划](../repro_archive/20260910/local_reencode/PLAN.md)。

5460张Train拟合、256张内部审计，固定2400步；末步唯一导出，完整1429张Val一次评估，不访问Test。C须对R2 IoU至少+0.003、对B至少+0.001且配对区间均为正，同时满足Dice、precision、小面积组及Brier非退化门槛，才允许继续正式结构研究。历史最好EPPA仍作为绝对Test参考，不能只声称超过C4就解决问题。

**已完成，未通过推进门：** 原R2/B现有特征重解码/C原图重编码的Val macro IoU为72.6654%/72.6077%/72.6248%，Dice为82.4034%/82.3541%/82.3632%。C对R2 IoU−0.0405个百分点，95%区间[−0.1142,+0.0331]；对B仅+0.0171个百分点，区间[−0.0123,+0.0459]，均跨0。9个条件仅小面积组recall不降通过，不能宣称增益或显著退化。固定拟合子集loss下降，但Train内部审计和Val IoU均无改善；Val中更多FP伴随更少FN，最小GT总面积组IoU−0.3732个百分点。停止该冻结纠错配置的80轮扩展，不访问Test，不将其称为第二创新点。[完整结果与诊断](../repro_archive/20260910/local_reencode/results/README.md)。

来源 `c749b72db327cd777f63a4b6da4583b301456538`，tag `diagnostic-r2-local-reencode-20260910`。主干状态哈希不变、1429张Val原R2逐图指标差值为0；本地复算summary、8425行计数指标及三组文件名互斥均通过。悉尼15:11:02.919结束、用户15:23:24.172触发唯一末检，间隔741.254秒，检查1/2，符合30分钟要求，无持续SSH。原heartbeat已不存在，删除接口确认为not_found，不把人工触发误记为自动检查。[执行证据](../repro_archive/20260910/local_reencode/execution/README.md)。

两组分支权重、逐图结果、冻结源码及执行材料共27文件、5,748,879逻辑字节已新增至HF Bucket `razaxq/BetterLViT` 的 `c749b72d/local_reencode_v1/`，逐文件大小和Xet哈希核验通过；既有模型未覆盖或删除。[上传核验](../repro_archive/20260910/local_reencode/execution/hf_upload_verified.json)。

## R2 局部语义分歧诊断（2026-09-10）

用户同意先验证内容自适应局部语义编码的机制假设。已预注册冻结R2 seed1219 Best的Train/Val诊断：512张Train拟合三个小型读出头，128张Train内部审计，完整1429张Val比较分歧/不确定性/纹理/随机选择；每张固定40/196块、共享纠错头。主干、文本、原分割头与原检查点保持冻结，无Test访问。定位、实际纠错和相对不确定性的价值三类门槛须全部通过才进入结构设计；诊断头不是第二创新，也不代表正式模块结果。[预注册计划与复现代码](../repro_archive/20260910/semantic_probe/PLAN.md)。

**诊断完成，未通过推进门：** 相同20.4082%像素预算下，分歧/不确定性/随机的每图错误捕获率均值为17.2916%/91.0813%/20.4823%；分歧规则实际纠错Val IoU为72.6599%，原R2为72.6654%，变化−0.0055个百分点，95%区间[−0.0210,+0.0105]跨零。9个预注册条件全部失败，不启动该规则的80轮结构实验。两条线性读出本身较弱（Val IoU25.3789%/21.7849%），只能否定本次分歧构造的证据，不能据此否定全部局部编码方向；Oracle不是可实现模型成绩。[完整结果与限制](../repro_archive/20260910/semantic_probe/results/README.md)。

诊断来源 `5f10bb1c4d83372553f204492ffcf3bbfaab782e`，tag `diagnostic-r2-semantic-probe-20260910`；原R2训练来源为 `9eca26de5b301099805530edbf5a1a8718bea662`。6项本地检查与两次真实Train批次8步CUDA预检通过；全流程主干状态哈希不变，1429张Val基线逐图差值为0。独立复算summary及5716项逐块选择/错误捕获核验通过。悉尼05:36:30完成、05:45:34唯一末检，相隔544秒，检查1/2，heartbeat `r2` 已删除，无持续SSH。未访问Test；既有模型保留。[启动与检查证据](../repro_archive/20260910/semantic_probe/execution/README.md)。诊断权重、结果、冻结源码与执行材料共24文件、15,142,978逻辑字节已添加至HF Bucket `razaxq/BetterLViT` 的 `5f10bb1c/semantic_probe_v1/`，逐文件大小和Xet哈希验证通过。[云端核验](../repro_archive/20260910/semantic_probe/execution/hf_upload_verified.json)。

## 最新 Test 结果（2026-09-10）

**项目历史水平的解释更正：** 用户指出R2对旧V4-B没有明显提升。现有证据支持“R2相对配对C4改善”，尚不支持“R2明显超过历史最强EPPA”。旧V4-B固定0.5的Test macro Dice/IoU为84.4562%/75.9546%，Val选阈值0.52后为84.4996%/76.0273%；R2三种子均值（固定0.5）为84.5031%/75.9833%。R2均值对旧V4-B固定0.5仅+0.0469/+0.0287个百分点，对旧校准成绩为+0.0035/−0.0440个百分点。历史A4（LoRA+EPPA）固定0.5为84.4932%/75.9909%，校准0.516为84.5191%/76.0446%。这些是历史分数参考，旧单次结果与R2多种子均值、不同阈值及训练配置不能当成配对消融。

R2的+0.5762个百分点IoU仍是相对C4的三种子配对结论，不追溯更改该实验的预注册判定；它不代表刷新历史最佳，也不构成第二结构创新。后续需同时报告项目历史参考与同训练配方/同种子/同阈值策略的强基线对照，明确是否取得超过既有EPPA水平的实际进展。当前无LoRA约束保持有效，不能为比较而静默改变配置。[旧V4-B原始记录](D:/BetterLViT/BetterLViT/BetterLViT_完整迁移包_2026-08-16/04_模型与结果/EXPERIMENT_RESULTS.md)、[R2三种子结果](../repro_archive/20260910/recipe_test/results/README.md)。

**最终归档完成：** 九组训练归档47文件，加本轮Test产物15文件，总计62文件、15,421,689,106逻辑字节，全部按Git哈希分目录上传HF并校验大小/Xet哈希。17个不同的服务器模型原文件仍存在，大小一致；系统盘余4,005,224,448字节，共享fs18,559,782,256字节，低于20GB硬上限。Test在04:44:16.998（悉尼）结束，05:00:19.107检查，距结束962.110秒，Test检查1次，无新增训练检查。全部完成后已删除本轮heartbeat。新凭据保存在标准HF缓存，Git中不含token。[HF完整核验](../repro_archive/20260910/hf_recovery/results/README.md)。

R2与C4三个匹配种子的六个Test评估全部完成，各2113张、固定阈值0.5、Val IoU选Best。平均IoU配对差值 **+0.5762个百分点**（样本SD 0.3961），平均Dice差值 **+0.3784个百分点**；三个IoU差值全部正向：True。[全部结果、来源与限制](../repro_archive/20260910/recipe_test/results/README.md)。R2为训练配方成果，不作为第二项结构创新；没有新增训练。

## 最新执行记录（2026-09-10）

**九组训练HF备份完成：** R1/R2、四组新增配对种子、P11的80/150轮和P12，共47文件、15,415,489,746逻辑字节已全部云端核验。P11续训Best从已验证父提交复制，Last来自续训提交，并保留runtime/selection来源记录。仅添加文件，未删除本地/服务器模型；新凭据不在源码或结果中。[逐文件核验](../repro_archive/20260910/hf_recovery/results/README.md)。本轮新Test输出仍按05:00预约确认后另行追加。

**HF认证与首批备份已恢复：** 新凭据已验证并更新标准本地缓存。六组R1/R2及配对种子训练的30个文件、10,156,614,282逻辑字节已全部上传至各自Git短哈希目录，源文件新鲜Xet哈希、服务器云端列表及独立本地列表全部一致。补查发现P11的80/150轮及P12未备份，正在补传其17个文件；新Test产物待预约确认完成后追加。模型保留，凭据不进入Git。[上传恢复与核验记录](../repro_archive/20260910/hf_recovery/README.md)。

**Test已提交：** 2026-09-10 04:29:26.558（悉尼）提交六模型顺序评估，PID182805；完整评估源码 `285785429e9584797a4130611c4e206244652533`，tag `evaluation-r2-recipe-test-3seeds-20260910`，已推送GitHub并核对六个训练标签。当前仅提交回执，无本轮Test结果。预测04:47:37.777完成，同一heartbeat已约05:00末检，Test预算0/2，不增加训练检查。启动前系统盘余4,024,750,080字节，共享fs为18,559,782,256字节，模型保留。[启动、协议及预约](../repro_archive/20260910/recipe_test/README.md)。

**三种子Val复验完成：** C4/R2的1219、2027、3407六组80轮训练全部结束。R2相对同种子C4的IoU增益分别+0.3857、+0.6291、+0.6170个百分点，平均 **+0.5440个百分点**（样本SD0.1372）；Dice平均+0.3873个百分点。三种子IoU均正、均值≥0.003、平均Dice与最小组IoU不降，注册复验条件全部通过。最后R2-3407 Best75，Val IoU/Dice为0.726484/0.823421；训练03:34:59结束，03:46:44末检，距结束705.476秒，检查2/2。本轮Test尚待执行，不能宣称稳定Test增益。[完整Val结果](../repro_archive/20260909/recipe_replication/THREE_SEED_SUMMARY.md)。

**Test阶段已准备：** 六个来源SHA及Val选定Best固定，统一2113张、阈值0.5、batch16，历史C4-1219也重新用同一导出器评估。预计后台顺序评估约18.2分钟，以提交回执预测完成后12分钟预约检查，不增加训练检查。[Test执行协议](../repro_archive/20260910/recipe_test/README.md)。R2是训练配方优化，第二项结构创新仍未取得可靠Test增益。

## 最新执行记录（2026-09-09）

**R2-3407首次检查：** 23:03:41.164（悉尼）的第1次短连接确认最后一组训练健康，完成4/80轮、正在第5轮，SHA/manifest一致、跟踪文件干净、未访问Test。第2–4轮整轮均值214.507214秒，预计2026-09-10 03:33:39.592结束；同一heartbeat已改约9月10日03:46末检，独立预算1/2，之后不再中途连接。最终按真实结束时间核实30分钟约束，完成后再作3407配对及三种子汇总。[首检、预测和预约归档](../repro_archive/20260909/recipe_replication/README.md)。

**R2-3407已提交：** C4-3407结果归档推送后，22:47:10.844（悉尼）提交最后一组，PID171080，SHA `7b498ba94fed230731d1daf9a5f4de042e4e71f2`，tag `experiment-r2-recipe-80e-seed3407-20260909`；仅有启动回执，尚未确认健康，独立预算0/2。同一heartbeat已改约23:03首检，再预测末检。保持注册配方，启动前系统盘余5,737,332,736字节，模型保留；本组完成后汇总全部三个配对再决定固定协议Test。[启动与预约归档](../repro_archive/20260909/recipe_replication/README.md)。

**C4-3407完成：** 80轮、Best79，1429张Val macro IoU **0.720314**、Dice **0.818852**，固定阈值0.5，未访问Test。训练22:32:01.329结束，Val导出22:33:51.216，末检22:43:51.462（悉尼），间隔710.132秒，检查2/2。原始日志、逐图记录、80轮实际LR及Best/Last来源均核验。三个种子的控制均完成，按注册队列接续R2-3407；第三个配对和三种子最终判断仍待完成。[完整结果](../repro_archive/20260909/recipe_replication/c4s3407_results/README.md)。

**C4-3407首次检查：** 18:00:35.268（悉尼）的第1次短连接确认训练健康，完成4/80轮、正在第5轮，SHA/manifest一致、跟踪文件干净、未访问Test。第2–4轮整轮均值214.745656秒，预计今天22:30:55.545结束；同一heartbeat已改约22:43末检，独立预算1/2，之后不再中途连接。最终按真实结束时间核实30分钟约束，当前没有3407最终结果。[首检、预测和预约归档](../repro_archive/20260909/recipe_replication/README.md)。

**C4-3407已提交：** R2-2027结果归档并推送后，17:44:10.824（悉尼）提交第三种子的C4对照，PID160309，SHA `02b28b2d455efdec4249cedd3d0ac7a865808283`，tag `experiment-c4-recipe-80e-seed3407-20260909`；仅有回执，尚未确认健康，独立预算0/2。同一heartbeat已改约18:00首检，再预测末检。保持注册配方，启动前系统盘余7,449,919,488字节；R2-3407已准备、未提交。当前没有3407最终结果或新增Test访问。[启动与预约归档](../repro_archive/20260909/recipe_replication/README.md)。

**R2-2027完成：** 80轮、Best68，1429张Val macro IoU **0.730481**、Dice **0.827094**，相对C4-2027 IoU **+0.006291（+0.6291个百分点）**，95%逐图配对区间[0.003035,0.009478]；Dice+0.004229、最小组IoU+0.008817，四项门槛均通过。Precision+0.017216、Recall−0.013681，需同时报告。训练17:29:50.586结束，Val导出17:31:32.500，末检17:40:33.807（悉尼），间隔643.221秒，检查2/2。来源/实际80轮LR/日志/逐图记录/Best/Last均核验。现有1219、2027两个配对IoU均正向，但三种子复验未完成，接续注册3407两组，当前不访问Test。[本次结果](../repro_archive/20260909/recipe_replication/r2s2027_results/README.md)；[两种子进展](../repro_archive/20260909/recipe_replication/TWO_SEED_PROGRESS.md)。

**R2-2027首次检查：** 12:56:37.379（悉尼）的第1次短连接确认训练健康，完成4/80轮、正在第5轮，SHA/manifest一致、跟踪文件干净、未访问Test。第2–4轮整轮均值214.928532秒，预计今天17:27:38.861结束；同一heartbeat已改约17:40末检，独立预算1/2，之后不再中途连接。最终按真实结束时间核实30分钟约束，当前没有本种子的R2最终结果。[首检、预测和预约归档](../repro_archive/20260909/recipe_replication/README.md)。

**R2-2027已提交：** 完成C4-2027归档并推送后，12:40:37.904（悉尼）提交匹配R2训练，PID149524，SHA `c972a152b5d152ddfa82e034596085aca85dd0ab`，tag `experiment-r2-recipe-80e-seed2027-20260909`；只有回执，尚未确认健康，独立检查预算0/2。同一heartbeat已改约12:56首检，再预测末检；保持注册配方，不根据C4数值改配置。启动前系统盘余9,163,235,328字节。当前没有R2-2027最终结果或新增Test访问。[启动和预约归档](../repro_archive/20260909/recipe_replication/README.md)。

**C4-2027完成：** 80轮、Best70，1429张Val macro IoU **0.724190**、Dice **0.822864**，固定阈值0.5，未访问Test。训练12:27:33（悉尼）结束，Val导出12:29:31完成，最终检查12:37:40，间隔607.068秒（约10分钟），检查2/2；原始日志、逐图结果、80轮LR及Best/Last来源全部核验。这里只记录seed2027对照，待R2-2027完成再作同种子配对；接续原注册队列。[完整结果与来源](../repro_archive/20260909/recipe_replication/c4s2027_results/README.md)。

**C4-2027首次检查：** 07:54:39（悉尼）的第1次短连接确认训练健康，完成4/80轮、正在第5轮，SHA/manifest一致、跟踪文件干净、未访问Test。第2–4轮整轮均值214.800541秒，预计今天12:24:59.974结束；同一heartbeat `betterlvit-r2` 已改约12:37末检，预算1/2，此后不再中途连接。30分钟约束待末次按实际结束时间核实；当前无本种子最终结果。[首检、预测和预约归档](../repro_archive/20260909/recipe_replication/README.md)。

**配对种子复验首组已提交：** 四个冻结来源各两次CUDA预检全部通过，种子内C4/R2初始模型、输入和五步结果完全一致。C4-2027于07:38:08（悉尼）提交，PID138765，SHA `965382b24ad26de81b89f9ddfe9ab434ae4d105b`，tag `experiment-c4-recipe-80e-seed2027-20260909`；只有启动回执，尚未确认健康，独立预算0/2。新定时 `betterlvit-r2` 已约今天07:54首检，再按实测轮时预测末检。其余三组准备完毕、未提交；固定顺序C4-2027→R2-2027→C4-3407→R2-3407，完整跑完两个配对后按[预注册计划](RECIPE_REPLICATION_PLAN_20260909.md)汇总，当前不访问Test。[源码、预检、启动和预约归档](../repro_archive/20260909/recipe_replication/README.md)。

**R2筛选完成：** 80轮、Best67，Val macro IoU **0.726654**、Dice **0.824034**；相对C4 IoU +0.003857（+0.3857个百分点），95%配对图像区间[0.000512,+0.007145]，最小病灶组IoU +0.007661，Dice点估计不降，四项预注册门槛均通过。Precision −0.009615、Recall +0.015841；Dice区间仍跨0。当前为单种子Val候选，不是稳定Test成果。按已授权计划补充2027/3407的C4↔R2匹配训练，保持原配方，不组合R1或延长150轮。R2末检距训练结束839.206秒，预算2/2；旧R1/R2定时已删除。Best/Last和HF源文件清单已核对，认证旧问题未重试，模型保留。[两组汇总](../repro_archive/20260908/recipe_execution/SCREENING_SUMMARY.md)。

**R2首次检查：** 02:37的第1次短连接确认R2训练健康，完成4/80轮、正在第5轮，SHA/manifest一致、跟踪文件干净、未访问Test。第2–4轮整轮均值215.757秒，预计今天07:09:33（悉尼）完成；同一heartbeat已改约07:22进行最后一次检查。预算1/2，之后不再中途连接；完成预测及30分钟约束将在末次按实际时间核实。当前无R2最终结果，HF认证旧问题未重试。[快照和预测](../repro_archive/20260908/recipe_execution/README.md)。

**R2已提交：** 悉尼时间02:21:27发起R2后台训练，PID125897，来源 `9eca26de5b301099805530edbf5a1a8718bea662`，tag `experiment-r2-recipe-80e-seed1219-20260908-v2`。保持原增强和C4架构，仅采用注册的80轮单次余弦LR；没有根据R1数值改配置。当前只有启动回执，尚未检查健康，独立预算0/2；同一heartbeat已改到02:37首次检查，再据完整轮时预测末检。R1的HF认证问题单独待重新登录处理，模型均保留。[R2启动回执](../repro_archive/20260908/recipe_execution/r2_launch.json)。

R1完成80轮，Best78，Val macro IoU **0.723869**、Dice **0.822398**。相对C4 IoU仅+0.001073（+0.1073个百分点），95%配对图像区间[−0.003060,+0.005253]；未通过≥+0.003及正区间筛选。最小病灶组IoU +0.008239，第二面积组−0.008565；Precision下降、Recall上升，不据单一改善分组宣称成功。两次检查已用完，末次距训练结束约12分钟，符合约定。本轮未访问Test，停止R1扩展，按预注册计划接续独立R2。

R1来源 `f71e782940dc2a52b08111771f6e3d5f5a8d36c5`、80轮LR、逐图结果、原始日志和Best/Last来源均核对。HF凭据失效，本地认证与远端写入均401；五个待传文件的Xet清单及完整模型保留，重新登录后续传，不宣称上传已完成。系统盘有约12.6GB可用，R2接续不会依赖删除这些模型。[完整R1结果](../repro_archive/20260908/recipe_execution/r1_results/README.md)。

## 最新研究记录（2026-09-08）

**R1首次检查：** 21:29的第1次短连接确认R1训练健康，已完成4/80轮、正在第5轮，SHA与运行manifest一致，跟踪文件干净，未访问Test。第2–4轮整轮均值211.492秒，预计2026-09-09 01:56:03（悉尼）完成训练；同一heartbeat已预约02:09进行第2次也是最后一次检查。当前检查预算1/2，之后不再中途连接；最终必须核对真实结束时间与30分钟约束。R2仍未启动，当前没有完整新Val/Test结果。[预测与快照](../repro_archive/20260908/recipe_execution/README.md)。

**R1/R2配方实验：** 用户已授权按[正式实验计划](TRAINING_RECIPE_PLAN_20260908.md)执行。R1删除原90度倍数旋转/镜像分支并保留随机数消耗，R2把原10轮LR重启改为首轮3e-4至第80轮1e-6的单次余弦；分别与C4独立对照，均从头80轮、seed1219、batch16、224、原Dice/Focal、无LoRA/新监督/外部视觉分支。R1已于21:13:38提交后台（PID114892），仅有启动回执，尚未检查健康；R2准备完成、未启动。正式来源分别为 `f71e782940dc2a52b08111771f6e3d5f5a8d36c5` / `9eca26de5b301099805530edbf5a1a8718bea662`，tag分别为 `experiment-r1-recipe-80e-seed1219-20260908-v2` / `experiment-r2-recipe-80e-seed1219-20260908-v2`。9项回归检查通过，最终来源各两次真实Train批次五步CUDA结果一致，R2还与C4五步完全一致。R1检查预算0/2，首检21:29，再按实际整轮耗时预约末检；R1完成后继续R2，不以R1数值通过为前提。当前无新Val/Test结果。完成后按IoU≥+0.003、配对区间及非退化门槛筛选，再做匹配种子和最终Test；不把配方优化当第二项结构创新。完整来源、预检、启动与定时记录见[执行归档](../repro_archive/20260908/recipe_execution/README.md)。

**P12后复盘：** 已完成[训练策略复盘与数据审计](../repro_archive/20260908/strategy_review/复盘与下一步.md)。全部1429张Val mask及抽查32张Train mask本来已是224×224；当前resize与最近邻完全一致，排除这一步新增掩码边界变化。共用增强包含90度倍数旋转/镜像而报告token不变；P12七次学习率重启均伴随暂时Val回落，但两点都尚未被证明是最终退化原因。下一步建议限定为C4的胸片方向增强、单次学习率衰减两个独立对照，保持架构和原Dice/Focal，再按配对种子和最终Test验证。该复盘阶段只进行了研究/审计；后续启动状态以上方R1/R2记录为准，尚无新增Test访问。

**执行更新：** P12的80轮训练及自动Val导出已完成，Best epoch70，Val macro IoU/Dice为 **0.716820/0.816282**；C4为0.722796/0.821214。IoU下降0.005977（−0.598个百分点），配对图像95%区间[−0.010325, −0.001710]，最小面积组IoU下降0.008938，四项门槛均失败。停止当前中层相加配置，不启动已准备的C9、不扩150轮、不访问Test。检查总计2次，末次距实际训练结束约11分36秒，定时已删除。源码`5d09913d46863073cba93159af4ed61f88fdd96e`，完整证据见[P12最终结果](../repro_archive/20260908/visual_prior_execution/p12_results/README.md)。前置CXformer探针IoU0.530981优于同构DINOv2的0.475959，但该优势没有转化为本配置的完整BetterLViT增益。

P11 已完成用户指定的150轮，最佳仍为80轮，Test macro IoU/Dice为0.755661/0.842335；结果与原80轮Best一致。完整运行与逐图证据见[最终归档](../repro_archive/20260908/dual_grain_150_results/README.md)。延长训练没有带来增益。

针对“深入研究第二点方向”，已完成[冻结视觉预训练特征研究报告](../repro_archive/20260908/frozen_visual_prior/研究报告.md)。优先候选为CXformer-S，同结构自然图像DINOv2-S作预训练来源对照，DINOv3-S保留后续候选。计划从C4接入单处中层残差投影，约41,280个可训练参数，沿用原Dice/Focal；这是普通融合可行性方案，尚非独立创新或性能结果。报告记录直方图均衡化/分辨率混淆、固定权重版本、解析预算及Train/Val→配对种子→Test流程。

研究报告完成时仅进行了公开元数据审计；后续诊断与训练推进以上方执行更新为准。当前P12已得到完整BetterLViT单种子Val退化结果，无正增益或新增Test证据。

## 已完成的正式 Test 实验

所有指标均为 Test macro，样本数均为 2,113。

| ID | 配置 | 完整 Git SHA | 最佳 epoch | Dice/IoU @ 0.5 | 验证集选阈值 | Dice/IoU @ 选定阈值 | 结论 |
|---|---|---|---:|---:|---:|---:|---|
| B0 | Frozen CXR-BERT + original PLAM + Dice/BCE | `77f43ba26487c73b56da1080f83706eececb8203` | 120 | 0.835266 / 0.748076 | 0.656 | 0.840582 / 0.756348 | 正式完成 |
| A0 | CXR-BERT LoRA + original PLAM + Dice/BCE | `7e8fc4f9cc0602e98ce12dadcbc6d55052c4c331` | 109 | 0.831570 / 0.743281 | 0.626 | 0.834767 / 0.748608 | 正式完成 |
| A1 | CXR-BERT LoRA + original PLAM + Dice/Focal | `3e46ab48e4eb7c077e789d6d9afc0158cda18a73` | 80 | 0.835233 / 0.747126 | 0.560 | 0.837367 / 0.751033 | 正式完成；历史 Focal 实验 |
| A2 | CXR-BERT LoRA + FAM-EPPA V4-B + Dice/BCE | `994094f989069a4d5b7263dc4ad45d333e9d0ede` | 80 | 0.844099 / 0.757901 | 0.550 | 0.844966 / 0.759453 | 非 Focal 已完成结果中的最佳者 |
| A3 | CXR-BERT LoRA + FMISeg-adapted fusion + Dice/BCE | `1c0b7dae263b27860a3fd672431d6968f7beec0c` | 80 | 0.836136 / 0.749682 | 0.540 | 0.836527 / 0.750792 | 原理适配，不是完整 FMISeg 复现 |
| A4 | CXR-BERT LoRA + FAM-EPPA V4-B + Dice/Focal | `a1d40d3a305a34abc0e96885fae68532007485b2` | 80 | 0.844932 / 0.759909 | 0.516 | 0.845191 / 0.760446 | 历史 LoRA+EPPA 单种子结果；历史 Focal 实验 |
| A5 | CXR-BERT LoRA + FMISeg-adapted fusion + Dice/Focal | `4ef4bf0616975e4ebbab37ad660d63ee99b61216` | 139 | 0.835314 / 0.746882 | 0.562 | 0.838524 / 0.752085 | 原理适配；历史 Focal 实验 |
| A6 | Frozen CXR-BERT + TCSR V1 + original PLAM + Dice/BCE | `01cd2f7501804f07e04ad4595de2fc9e2c511ede` | 139 | 0.834810 / 0.747394 | 0.556 | 0.836082 / 0.749756 | LoRA=False；V1 未产生正增益 |
| A7 | Frozen CXR-BERT + TCSR V1 + FAM-EPPA V4-B + Dice/BCE | `75b6ed284f8e137fa6424040f578b8bc7afd1c5c` | 80 | 0.843212 / 0.757207 | 0.508 | 0.843273 / 0.757351 | LoRA=False；未形成稳健增益 |
| A8 | Frozen CXR-BERT + TCSR V2 + FAM-EPPA V4-B + Dice/Focal | `3e00c9016023f23fce690586150d6d524b3d6ecf` | 80 | 0.843154 / 0.757541 | 0.498 | 0.843147 / 0.757523 | 相对 A9 无稳健增益；历史 Focal 实验 |
| A9 | Frozen CXR-BERT + FAM-EPPA V4-B + Dice/Focal | `494ec30ce1cd94fac566b480877ea3391f5d64ed` | 150 | 0.842435 / 0.755412 | 0.548 | 0.843717 / 0.757799 | A4 的无 LoRA 配对；历史 Focal 对照 |

## TCSR validation-only pilot 结论

| Pilot | 机制 | 完整 Git SHA | 最佳 Val macro Dice/IoU @ 0.5 | 关键诊断 | 阶段结论 |
|---|---|---|---:|---|---|
| P1 / V2.1 | 边界保持非对称两跳路由 | `e00138529f5ce5ac3996f58e54f764274789081e` | 0.8177 / 0.7180 | 相对 A9 best Dice -0.0063；最小病灶 -0.0132；`x4→x3` gate 0.7047，`x3→x2` 关闭 | 失败；无 80 epoch/Test |
| P2 / V2.2 | `x3→x2` 单跳边界路由 | `58be4f092b4a1c2f141e510ff40766fb58d4519b` | 0.8153 / 0.7151 | 较 A9 epoch40 -0.0011；相对 A9 best -0.0088；最小病灶 -0.0206；gate 0.9898 饱和 | 失败；无 80 epoch/Test |
| P3 / V2.3 | 有界校准 gate | `874adf6b51e9e3a30b5295b36a1992b6c9098abb` | 0.8154 / 0.7149 | 较 A9 epoch40 -0.0009；相对 A9 best -0.0086；最小病灶 -0.0167；gate 0.3768，边界支持仍过密 | 失败；无 80 epoch/Test |
| P4 / V2.4 | 稀疏边界有界单跳路由 | `e07924b22956099207203f792eb3c36eb3dd11db` | 0.813003 / 0.712870 | 较 A9 epoch40 -0.003327；相对 A9 best -0.011044，95% CI [-0.014624, -0.007416]；最小病灶 -0.027197；gate 0.27998、focus 0.16654、delta RMS 0.00973；train-val gap约0.0054 | gate/稀疏性通过，但数值与小病灶失败；无 80 epoch/Test |

P4 验证导出：`/root/autodl-tmp/BetterLViT-paper-p4-tcsrv24/runtime_logs/p4_validation_acceptance/`。P1–P4 均未访问 Test。

## 当前非 Focal 配对验证：C0 → P5

### 锁定配置与溯源

| ID | 配置 | 正式 Git SHA | 标签 | GPU 预检 | 当前状态 |
|---|---|---|---|---|---|
| C0 | Frozen CXR-BERT + 无 TCSR + FAM-EPPA V4-B + Dice/Tversky；LoRA=False | `281bc1b40c782f6f42b8caa332e1fe1045cb29e0` | `pilot-c0-tversky-frozen-b16-seed1219-20260902` | 两次 batch16 整模输出一致；SHA-256 `8475bc883e5150be77eeff1541fd2afb32fa3ba1961f75ee06589ae75d1fd3a6`；峰值 allocated/reserved 15.079/16.441 GB | 40 epoch 状态 0；best epoch 29；validation macro Dice/IoU/precision = 0.810587/0.708856/0.824030；1429 样本；未访问 Test |
| P5 | 与 C0 相同，但启用 TCSR V2.5：`x3→x2` 单跳稀疏路由 + 训练期局部监督；LoRA=False | `7d3bfce1caf0444656abfa9e42531f99a495c539` | `pilot-p5-tcsrv25-tversky-frozen-b16-seed1219-20260902` | 模块 SHA-256 `3dc9354fbf4fd95761ac3ec9dd5658ac5faefd76def7bb3e6ac82b6c5e7bc060`；整模 SHA-256 `a20a520fdde460a1d5aca1ad0a3efd7a089a84a06b15fe120a00539549d1d407`；峰值 15.791/17.178 GB | 40 epoch 状态 0；best epoch 30；validation macro Dice/IoU/precision = 0.808773/0.706784/0.818106；数值门失败；其局部项包含边界监督，不符合最终方法约束；未访问 Test |

P5 局部监督：GT 边界 Dice 对齐 + `0.5 *` 边界外残差泄漏，总权重 0.02，前 5 epoch 线性 warmup；推理不需要标签。模块预检确认 V2.5 与 V2.4 推理误差为 0，重复误差为 0，`x1/x3/x4` 严格 identity，文本/跨尺度效应与所有必要梯度非零。该实验保留为诊断证据，但因使用了边界监督，不得成为最终方法或后续配置模板。

### 最终验证状态

- C0 和 P5 均已按 40 epochs、batch 16、seed 1219、deterministic、`drop_last=True` 完成，状态均为 0。
- 两份 best-checkpoint validation-only JSON 均核对通过：`split=validation`、`test_split_accessed=false`、1429 样本、提交来源一致且包含 precision。
- P5 相对 C0：macro Dice `-0.001814`，95% CI `[-0.004823, 0.001178]`；macro IoU `-0.002072`；macro precision `-0.005924`，95% CI `[-0.009139, -0.002782]`。
- 最小病灶四分位：Dice `+0.000351`，但 precision `-0.004655`，因此未满足“小病灶 Dice 与 precision 均不下降”。
- P5 gate `0.2583`，位于目标范围；`x1/x3/x4` identity 通过；边界外泄漏由训练中期约 `0.381` 降至 epoch 40 的 `0.280`；epoch 40 train-validation Dice gap 约 `0.0081`，未见异常。
- 结论：P5 数值门失败，不扩展到 80 epoch，不访问 Test；当前无活跃训练。

### P5 阶段门

P5 只与相同损失、seed、batch 和训练长度的 C0 比较，必须同时满足：

1. validation macro Dice 相对 C0 至少 `+0.002`；
2. 最小病灶四分位 Dice 不下降；
3. 最小病灶四分位 precision 不下降；
4. `x1/x3/x4` 保持严格 identity；
5. route gate 位于 `[0.15, 0.35]`；
6. 边界外残差泄漏随训练改善；
7. 无异常 train-validation gap。

任一关键项失败即停止，不扩展到 80 epoch，也不访问 Test。即使通过，也先记录并等待多 seed 正式计划。

## 运行与结果路径

- C0 runtime：`/root/autodl-tmp/BetterLViT-paper-c0-tversky/runtime_logs/`
- P5 runtime：`/root/autodl-tmp/BetterLViT-paper-p5-tcsrv25/runtime_logs/`
- 非 Focal GPU 预检：`/root/autodl-tmp/BetterLViT-paper-p5-tcsrv25/runtime_logs/preflight_20260902_nonfocal/`
- C0 完成后 validation 导出：`runtime_logs/c0_validation_acceptance/c0_best_val_0p5.json`
- P5 完成后 validation 导出与 C0 配对比较：记录在 P5 的 `runtime_logs/p5_validation_acceptance/`

## 决策日志

- 2026-09-02：C0/P5 配对阶段采用 Dice/Tversky，并以 FP 0.7、FN 0.3 强化假阳性惩罚。
- 2026-09-02：P4 失败；停止沿现有 TCSR 路由继续参数修补。
- 2026-09-02：C0/P5 完成确定性 GPU 预检、独立提交与标签锁定；C0 启动。
- 2026-09-02：停止维护 `改动计划.xlsx`，后续只更新本 Markdown 台账。
- 2026-09-02：C0 以状态 0 完成；best epoch 29 的 validation macro Dice/IoU/precision 为 0.810587/0.708856/0.824030，未访问 Test。随后按门控链启动 P5。
- 2026-09-02：P5 以状态 0 完成，但相对 C0 的 macro Dice 为 -0.001814，最小病灶 precision 为 -0.004655；尽管 gate、identity、泄漏改善和 train-validation gap 通过，数值门仍失败。停止 P5，不进行 80 epoch 或 Test。
- 2026-09-02：用户澄清 Focal 可以使用，禁止的是 boundary loss。后续恢复 Dice/Focal 候选，所有正式配置保持 `boundary_loss=0.0`；P5 因包含边界监督仅作为历史诊断实验。
- 2026-09-03：P6 的频率/分支分层诊断确认：粗头训练显著干扰共享主干，最终 residual 虽能救回大部分损失，却主要退化为全局负向置信度收缩；高局部纹理样本的救援幅度最弱。因此停止 BCDH-R 参数修补，转向梯度隔离、局部稀疏且正负平衡的 CDRR。
- 2026-09-03：锁定并启动 C2→P7 的 40-epoch validation-only 配对。两项均为 Frozen CXR-BERT、FAM-EPPA V4-B、Dice/Focal、LoRA=False、`boundary_loss=0.0`，明确禁止 Test；C2 为严格控制，P7 仅增加 CDRR V1。
- 2026-09-03：共享目录必须保持在 20 GB 以下。完成存储审计和裁剪后，`betterlvit_5090_migration` 从 24,266,917,767 bytes 降至 18,559,777,853 bytes；训练盘可用空间从 4.9 GB 提升至 7.3 GB。后续不得再把完整训练会话无条件复制到共享盘。
- 2026-09-03：C2/P7 均完成 40 epochs、best validation 导出和配对比较，状态为 `complete`，未访问 Test。P7 相对 C2 的 macro Dice 显著下降 `-0.003694`，小病灶 Dice 下降 `-0.008555`，两个高频口径也未同时满足 Dice/precision 不下降；CDRR V1 数值门失败，不扩展训练。

## 研究反思与主线收敛

### 已被证据支持的结论

- FAM-EPPA V4-B 是目前最可信的架构贡献。A2 相对 A0、A4 相对 A1 均获得约 0.8–1.0 个百分点的 Test macro Dice 增益；在冻结文本编码器的 A6/A7 配对中，加入 FAM-EPPA 的 A7 也明显优于 original PLAM 的 A6。
- LoRA 不是必要增益来源：A0 低于 B0，冻结文本编码器的 A9 仍接近历史最佳。因此后续保持 Frozen CXR-BERT、LoRA=False。
- FMISeg 原理适配未形成可靠增益：A3/A5 均低于对应的 FAM-EPPA 实验；不能把局部借鉴描述成完整 FMISeg 复现或第二项已验证创新。
- TCSR V1–V2.5 尚不能作为性能型核心创新。P1–P4 依次修复路由关闭、gate 饱和、支持区域过密等机制问题；P5 又加入直接局部监督并换用 Dice/Tversky。最终 gate、identity、泄漏下降和泛化间隙均正常，但相对严格 C0 对照仍降低 macro Dice 和 precision。
- P5 的结果说明，即使加入额外局部监督，文本条件跳连路由仍未形成数值增益。当前路由与已有 FAM-EPPA/PLAM 的作用存在冗余或干扰，且更容易扩大预测区域、降低 precision；继续做 TCSR V2.6 式参数修补的研究价值很低。由于最终方法禁止 boundary loss，P5 的局部监督也不能继续沿用。

### 当前论文论点边界

1. 第一项架构创新可以使用 FAM-EPPA：自适应频率感知的解码器融合，并由多组配对结果支撑。
2. TCSR 目前只能作为系统探索及负结果，不能作为第二项已成立的核心创新。
3. Focal 可以作为主线损失；禁止 boundary loss。论文主口径继续使用逐样本 macro Dice/IoU。

### 建议的第二项创新转向（尚未开发）

停止继续修改文本路由，改为验证不需要边界目标的 **Cross-Scale Prediction Agreement Refiner（暂名 CPAR）**：

- 保留 Frozen CXR-BERT、LoRA=False、FAM-EPPA V4-B，恢复 Dice/Focal，明确 `boundary_loss=0.0`，不再启用 TCSR。
- 在最终分割头之外增加一个来自较粗解码尺度的轻量辅助分割头；两个头都只使用原始 segmentation mask 和 Dice/Focal，不生成边界图、距离图或边界损失。
- 使用粗、细两个预测的差异图作为不确定区域提示，让粗尺度上下文抑制孤立假阳性、细尺度分支保留局部结构。
- 修正分支采用零初始化、幅度受限的 residual logits；初始化时必须严格等价于无 CPAR 基线。
- 可加入预测一致性约束，但不能包含任何 boundary target/loss。推理阶段只保留主头与轻量修正，不需要 GT。
- 先建立相同 Dice/Focal、boundary loss 0 的 40-epoch validation-only 对照，再做 CPAR 配对；仍以 macro Dice `+0.002`、整体及最小病灶 precision 不下降为阶段门。通过后再做多 seed，最后才允许正式 Test。

该方向的论文组合将是：**FAM-EPPA（频率感知融合） + CPAR（跨尺度预测一致性修正）**。两者分别解决特征融合和输出校准，不依赖被禁止的 boundary loss，也比 EPPA + TCSR 更正交。

### BCDH 深入研究后的收敛设计（尚未开发）

CPAR 已进一步具体化为 **BCDH-R V1（Boundary-Conscious Dual-Head Refiner）**。完整论证、公式、代码接入点、排除项与实验门见 [`BCDH_RESEARCH.md`](BCDH_RESEARCH.md)。

- 不使用“分割头 + 边界头”，而使用 `112×112` 粗尺度完整 mask 头和 `224×224` 细尺度完整 mask 头；两头都只接受原始 segmentation mask 的 Dice/Focal 监督。
- 从粗、细预测本身生成 uncertainty、fine-only disagreement 和 coarse-only disagreement；不生成 GT boundary/distance/direction target。
- 在最终 logits 上使用零初始化、幅度受限的稠密 residual correction；初始化严格等价无 BCDH 基线。
- 建议首轮为 C1（A9 式严格控制）对 P6（BCDH-R V1）的 40 epoch validation-only 配对，batch16、seed1219、`boundary_loss=0.0`，不访问 Test。
- 阶段门除 macro Dice `+0.002` 外，还要求整体及最小病灶 precision 不下降、boundary F1 提升、Brier score 不恶化，并排除 residual 饱和或全图修正。
- “双头”或普通 deep supervision 本身不是充分创新；论文论点必须落在 prediction-only 的有符号跨尺度误差提示与 exact-identity bounded residual correction 上。
- 在 boundary F1/Brier/ECE 证据出现前，C 不解释为 *Calibrated*，避免未经验证的概率校准主张。

## BCDH-R V1 实施与当前训练：C1 → P6

### 锁定实现

- 开发提交：`8a0fe9b41bc0a5d515a353d52693cb06afeb3096`。
- BCDH-R 使用 `up2` 粗尺度完整 mask 头、原有 `up1` 细尺度完整 mask 头、prediction-only uncertainty/disagreement cues，以及零初始化、`delta_max=1.0` 的 bounded logit residual。
- P6 目标：`L_total = L_Dice/Focal(final) + 0.2 * L_Dice/Focal(coarse)`；两项都只使用完整 segmentation mask。
- C1/P6 均为 Frozen CXR-BERT、LoRA=False、FAM-EPPA V4-B、Dice/Focal `0.5/0.5`、Focal gamma `2.0`、`boundary_loss_weight=0.0`。

### 独立提交、标签与预检

| ID | 完整 Git SHA | 标签 | 配置 | 预检 |
|---|---|---|---|---|
| C1 | `106aeab700ee98653b5ec27994a0aa15b60dac07` | `pilot-c1-bcdh-control-frozen-b16-seed1219-20260902` | BCDH 关闭的严格配对对照 | 两次 batch16 前/反向输出 SHA-256 均为 `d51177e6a9c6b162766de6c0e2eabfdeb4c62d10e815d601c3efdf30d9eb2c86`；loss 均为 0.2612988949；reserved 峰值 16.441 GiB |
| P6 | `7217660e6ef16e2a495bab4f20c73403468f55e1` | `pilot-p6-bcdh-r-v1-frozen-b16-seed1219-20260902` | C1 + BCDH-R V1 + 0.2 粗头辅助完整-mask监督 | 两次 batch16 前/反向输出 SHA-256 与 C1 完全一致；loss 均为 0.3125534058；identity error 0；reserved 峰值 16.852 GiB |

模块单测同时确认：零初始化 identity error 0、重复前向误差 0、粗头/输出投影/第二步 refiner trunk 梯度均非零，且未使用 boundary target。

### 当前运行状态

- 训练链于服务器时间 `2026-09-02T20:41:07+08:00` 启动，顺序为 C1 训练 → C1 best validation 导出 → P6 训练 → P6 best validation 导出 → 配对比较。
- 两项均为 40 epochs、batch 16、seed 1219、deterministic CUDA、`drop_last=True`。
- `AUTO_TEST_EVALUATE=0`、`TEST_SPLIT_ALLOWED=0`；本轮禁止访问 Test。
- 最终状态：C1 与 P6 均完成 40 epochs，训练、best validation 导出和配对比较均成功；GPU 已空闲。
- 运行元数据：`/root/autodl-tmp/BetterLViT-paper-p6-bcdh/runtime_logs/bcdh_pair_current.env`。
- 两份 validation JSON 均为 1429 样本、提交一致、`test_split_accessed=false`；本轮未访问 Test。

### 预注册阶段门

P6 相对 C1 必须同时满足：validation macro Dice 至少 `+0.002`、整体 precision 不下降、最小病灶四分位 Dice/precision 均不下降、tolerance-2 boundary F1 提升、Brier 不恶化，并且 residual 统计不出现饱和或全图无差别修正。未通过则停止，不扩展训练、不访问 Test。

### 最终结果与结论

| ID | Best epoch | Val macro Dice | IoU | Precision | Recall | Boundary F1 tol=2 | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1 | 40 | 0.816330 | 0.716653 | 0.785124 | 0.891991 | 0.726298 | 0.023294 |
| P6 | 40 | 0.815978 | 0.716806 | 0.784709 | 0.890255 | 0.725759 | 0.022994 |

P6 相对 C1：macro Dice `-0.000352`（95% CI `[-0.002388, 0.001636]`）、IoU `+0.000152`、precision `-0.000414`、recall `-0.001735`、boundary F1 `-0.000539`；Brier 改善 `-0.000300`（越低越好）。最小病灶四分位 Dice `-0.002482`、precision `-0.003837`、boundary F1 `-0.002898`。

数值门失败。最终 BCDH residual 的平均绝对幅度为 `0.523526`，`98.87%` 为负修正；高 uncertainty 区的修正幅度 `0.486574` 反而低于其余区域 `0.532763`，说明分支主要学成全局单向收缩，而不是预期的不确定区域局部校正。P6 不扩展至 80/150 epochs，也不访问 Test；BCDH-R V1 只能保留为负结果与后续机制诊断依据。

## BCDH 频率诊断与 CDRR V1：C2 → P7

### P6 分支与高频分层诊断

- 诊断工具提交：`5e7e708e5732cc455aa867e48a16962b0e007fce`；只读取 P6 best checkpoint 与 C1/P6 validation 结果，`training_performed=false`、`test_split_accessed=false`，样本数 1429。
- P6 base 相对 C1：macro Dice `-0.018400`、precision `-0.055832`、recall `+0.040846`、Brier `+0.002975`，说明粗头辅助梯度已明显改变共享特征并扩大预测区域。
- P6 coarse 相对 C1：macro Dice `-0.003289`、precision `-0.008741`；P6 final 相对 C1则为 Dice `-0.000352`、precision `-0.000414`、Brier `-0.000300`。final 相对 base 救回 Dice `+0.018047`、precision `+0.055418`，但没有越过控制组。
- final residual 平均 `delta_mean=-0.456584`，只有 `9.48%` 像素为正修正；top-20% uncertainty 的绝对修正反而比其余区域低 `0.016780`。该分支主要学成全图负向收缩，而非局部双向纠错。
- Haar 高频最高四分位上 final 相对 C1 的 Dice 为 `+0.002512`；但归一化局部细节最高四分位为 `-0.002740`，且 final 对 base 的救援仅 `+0.009033`，远低于最低局部细节四分位的 `+0.026507`。高频能量口径不一致，支持后续同时报告 Haar 与归一化局部细节，而不能只挑有利分层。
- 诊断结果：`/root/autodl-tmp/BetterLViT-paper-p6-bcdh/runtime_logs/bcdh_frequency_diagnostic/p6_frequency_heads.json`。

### CDRR V1 设计

CDRR（Cross-scale Detail Reliability Refiner）不使用边界标签或 boundary loss，针对上述失败机制做三项结构限制：

1. 粗头和 refiner 输入从 segmentation trunk `detach`，阻断辅助目标对主干的梯度干扰；主分割基线仍由原 Dice/Focal 独立优化。
2. 使用细/粗尺度局部细节一致性、预测 disagreement 与 base uncertainty 构造 reliability，只选择确定性的 top-15% 像素支持集；支持集外 residual 严格为 0。
3. 支持集内先做加权中心化，再施加 `0.5 * tanh` 有界 residual，使正负修正均可发生，避免再次退化为全局单向阈值偏移；末层零初始化保证初始预测与控制组逐位相同。

开发提交：`45a2a5a6fa89ca80e399eab377005399a7fb3833`。模块检查确认 identity/repeat error 均为 0、支持率 `0.149992`、支持集外最大 residual 为 0、主干梯度隔离通过、refiner/粗头/base 梯度非零，正/负活动像素占比约 `45.58% / 54.42%`，未使用 boundary target。

### 独立提交、标签与正式预检

| ID | 完整 Git SHA | 标签 | 配置 | 正式提交 GPU 预检 |
|---|---|---|---|---|
| C2 | `06479cd3302a8ca11022eac0a6b62bdad097eb65` | `pilot-c2-cdrr-control-frozen-b16-seed1219-20260903` | Frozen CXR-BERT + FAM-EPPA V4-B + Dice/Focal；CDRR 关闭 | 两次 batch16 前/反向输出 SHA-256 均为 `d51177e6a9c6b162766de6c0e2eabfdeb4c62d10e815d601c3efdf30d9eb2c86`；loss `0.2612988949`；allocated/reserved 峰值约 15.014/15.047 GiB |
| P7 | `fe4547a0c60fe948c9a574d9afc7d691370aeb42` | `pilot-p7-cdrr-v1-frozen-b16-seed1219-20260903` | C2 + CDRR V1 + 0.1 粗头完整-mask辅助监督 | 两次输出 SHA-256 与 C2 逐位相同；loss `0.2869261503`；identity error 0；allocated/reserved 峰值约 15.224/15.439 GiB |

两项均锁定 40 epochs、batch 16、seed 1219、deterministic、`drop_last=True`、LoRA=False、`boundary_loss_weight=0.0`。训练链设置 `AUTO_TEST_EVALUATE=0`、`TEST_SPLIT_ALLOWED=0`，只允许 best-checkpoint validation 导出与配对比较。

### 运行状态与阶段门

- C2→P7 链于服务器时间 `2026-09-03T12:27:26+08:00` 启动；运行元数据：`/root/autodl-tmp/BetterLViT-paper-p7-cdrr/runtime_logs/cdrr_pair_current.env`。
- 最终状态为 `complete`：C2/P7 均完成 40 epochs、best-checkpoint validation 导出与配对比较；GPU 已空闲，训练/导出/比较日志无错误。
- P7 相对 C2 的主阶段门：validation macro Dice 至少 `+0.002`；整体及最小病灶四分位 precision 不下降；tolerance-2 boundary F1 提升；Brier 不恶化；归一化局部细节最高四分位 Dice/precision 均不下降；支持率接近 15%，支持集外 residual 为 0，且修正不退化为单向全局偏移。
- 阶段门失败则停止，不扩展 80/150 epochs、不访问 Test；通过后也先做多 seed validation 复核，再决定是否进入正式 Test。

### C2/P7 最终结果

两份结果均核对为 `split=validation`、`test_split_accessed=false`、1429 样本、提交来源一致、LoRA=False、`boundary_loss_weight=0.0`。C2 best epoch 40；P7 best epoch 39。

| ID | Val macro Dice | IoU | Precision | Recall | Boundary F1 tol=2 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| C2 | 0.816330 | 0.716653 | 0.785124 | 0.891991 | 0.726298 | 0.023294 |
| P7 | 0.812636 | 0.711743 | 0.787725 | 0.881132 | 0.724445 | 0.023394 |

P7 相对 C2：macro Dice `-0.003694`，95% CI `[-0.006475, -0.000939]`；IoU `-0.004911`；precision `+0.002601`；recall `-0.010859`；boundary F1 `-0.001853`；Brier `+0.000100`（变差）。最小病灶四分位 Dice `-0.008555`、IoU `-0.010914`、precision `+0.011374`、recall `-0.035319`、boundary F1 `-0.009366`。

高频最高四分位结果并不稳健：

- Haar/Laplacian 能量口径：Dice `-0.001610`，precision `+0.009266`，未通过；
- 归一化局部细节口径：Dice `+0.000302`，precision `-0.000873`，未通过。

CDRR 机制约束本身工作正常：支持率 `0.149992`，支持集外 residual 最大值为 0，活动像素正/负修正约 `53.88% / 46.12%`，平均绝对 residual `0.021526`，未再次退化为全图单向收缩。但它通过提高 precision 换取了明显 recall 损失，尤其小病灶 recall 下降 `-0.035319`，导致整体 Dice、边界 F1 与 Brier 均未改善。

结论：CDRR V1 数值门失败，不扩展至 80/150 epochs，不做多 seed 或 Test。该结果说明“梯度隔离 + 稀疏双向修正”解决了 BCDH 的机制退化，却没有解决支持区域选择对真阳性细节的误抑制；后续如继续研究，必须先做离线错误归因或重新设计 reliability，而不是直接调节 residual 幅度或训练轮数。

### 历史结果归档

C1/P6 会话、清单、运行日志及频率诊断已按完整 Git SHA 归档到共享盘：

- `/root/autodl-fs/betterlvit_5090_migration/paper_experiment_artifacts/106aeab700ee98653b5ec27994a0aa15b60dac07/`
- `/root/autodl-fs/betterlvit_5090_migration/paper_experiment_artifacts/7217660e6ef16e2a495bab4f20c73403468f55e1/`

两项会话均核对为 2862 个文件、源/目标字节数完全一致，四个 Best/Last 检查点 SHA-256 全部匹配。服务器本地的两个旧会话副本已删除以释放训练盘空间，完整归档可恢复。

### 服务器存储策略（20 GB 共享目录上限）

- 2026-09-03 审计前：共享目录 24,266,917,767 bytes，本地训练盘仅余 4.9 GB。
- 已从 6 个 8 月旧会话中删除可由 sibling Best 替代的 Last checkpoint；8 个历史 Best 全部保留，另保留 3 个 Last。删除前的 SHA-256、字节数与完整路径记录在 `/root/autodl-fs/betterlvit_5090_migration/runtime_logs/storage_optimization_20260903_deleted_legacy_last.sha256`。
- 已删除共享盘中未被训练使用、可重建的旧 `huggingface-cache`；当前训练只使用 `root_cache/huggingface`。
- 已删除本地无活跃进程使用的 `hf-upload-staging`、`wheelhouse-cu128` 和旧 `huggingface` 缓存。上传暂存区不是唯一结果来源；wheelhouse/缓存可重建。
- 审计后：共享目录 18,559,777,853 bytes（约 18.56 GB），低于 20 GB；本地训练盘可用 7.3 GB，P7 训练保持正常。
- 后续保留顺序：结果 JSON、manifest、Git SHA/标签、Best checkpoint 优先；Last 只在需要断点续训时保留。完整会话优先发布到已校验的 Hugging Face Bucket 或外部归档，不再默认写入共享盘。每次归档前必须先检查 20 GB 上限。

## RACE-Fuse V1：C3 → P8（80 epochs）

### 研究动机与机制

P6/P7 证明单纯在输出端做局部修补容易以 recall 换 precision，尤其伤害小病灶。RACE-Fuse（Report-Anatomy Consistency and Evidence Fusion）改为在四层 encoder skip 上融合报告位置语义与视觉病灶证据：冻结 CXR-BERT 的 token 特征学习六区位置槽和三类病灶数量槽；每个尺度独立预测视觉 evidence；只有“报告正证据 × 局部视觉证据 × 图文一致性”共同成立时才开启有界加性路由。未提及区域按 unknown 处理，不作为病灶负标签；路由强度零初始化，未学习前与控制网络逐位相同。

训练数据的随机旋转/翻转会同步作用于六区解剖基底，避免图像增强后文本方位与空间先验错位。RACE 不使用 LoRA，主目标固定 Dice/Focal `0.5/0.5`，辅助权重 `0.05`，`boundary_loss_weight=0.0`。

### 独立提交、标签与预检

| ID | 完整 Git SHA | 标签 | 配置 | 正式提交 GPU 预检 |
|---|---|---|---|---|
| C3 | `424f2824b737e4d311cdcb270bbc83e973dc067f` | `pilot-c3-race-control-frozen-b16-seed1219-80e-20260903` | Frozen CXR-BERT + FAM-EPPA V4-B + Dice/Focal；RACE 关闭 | 两次 batch16 输出 SHA-256 均为 `d51177e6a9c6b162766de6c0e2eabfdeb4c62d10e815d601c3efdf30d9eb2c86`；loss `0.2612988949`；allocated/reserved 峰值约 15.10/16.46 GiB |
| P8 | `48df2f185a28919e38e073322809d043e176d382` | `pilot-p8-race-fuse-v1-frozen-b16-seed1219-80e-20260903-restart2` | C3 + RACE-Fuse V1；aux weight 0.05；四尺度最大 residual strength 0.15 | 最终提交两次 batch16 输出 SHA-256 均与 C3 逐位相同；loss `0.2993894517`（含辅助项）；identity strength 0；allocated/reserved 峰值约 17.33/18.70 GiB；真实训练样本增强/六区基底数据契约检查通过 |

早期 P8 提交 `d3ac4ad377447ed2fdb9d3176ae9369c106a77d4` 在首次 GPU 预检发现辅助 mask 多余通道维，未启动训练、无实验结果；`edb719c6245e2005853b220e233e9fcc76fae405` 修复后通过两次合成预检，但最终正式提交另加入真实 DataLoader 契约检查，因此两者均不得作为 P8 结果提交。

### 当前运行状态与阶段门

- C3→P8 验证专用链于服务器时间 `2026-09-03T18:30:26+08:00` 启动；元数据：`/root/autodl-tmp/BetterLViT-race-p8/runtime_logs/race_pair_current.env`。
- 两项均锁定 80 epochs、batch 16、seed 1219、deterministic、`drop_last=True`、LoRA=False、boundary loss=0；`AUTO_TEST_EVALUATE=0`、`TEST_SPLIT_ALLOWED=0`。
- 最终链状态为 `complete`：C3/P8 均完成 80 epochs，两个 best-checkpoint validation 导出和配对 bootstrap 比较均成功；GPU 已空闲，日志无训练、CUDA、数据或检查点错误。
- 两组 Best/Last 均可严格加载，`source_git_commit` 分别匹配上表提交；均为 LoRA=False、boundary loss=0，且 Test 从未访问。
- P8 相对 C3 的预注册门：macro Dice 至少 `+0.002`；整体 precision 不下降；最小病灶四分位 Dice 与 recall 不下降；Brier 不恶化；RACE 路由强度/门控/视觉 evidence/一致性统计不得坍缩或退化为全图均匀增强。模型选择完成前不访问 Test。

### C3/P8 最终结果

两份结果均为 `split=validation`、`test_split_accessed=false`、1429 样本、阈值 0.5，且 best epoch 都是 80。

| ID | Val macro Dice | IoU | Precision | Recall | Boundary F1 tol=2 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| C3 | 0.821214 | 0.722796 | 0.814701 | 0.864661 | 0.739182 | 0.021477 |
| P8 | 0.823061 | 0.725566 | 0.813476 | 0.869796 | 0.743003 | 0.021082 |

P8 相对 C3：macro Dice `+0.001848`，95% CI `[-0.001263, 0.005013]`；IoU `+0.002770`；precision `-0.001225`；recall `+0.005135`，95% CI `[0.000921, 0.009447]`；boundary F1 `+0.003821`；Brier 改善 `-0.000394`，95% CI `[-0.000628, -0.000159]`（越低越好）。

最小病灶四分位（358 样本）Dice `+0.001947`、precision `+0.001469`、recall `+0.007525`，但区间均跨 0；boundary F1 `-0.002337`。高频最高四分位在两种预注册口径均为正：Laplacian Dice/precision `+0.005027/+0.002907`，归一化局部细节 `+0.002645/+0.000626`。

RACE 末轮四路 strength 为 `[-0.05793, 0.06121, 0.06121, 0.08668]`，gate mean `0.3169–0.3592`，visual evidence mean `0.7187–0.7852`，agreement mean `0.8095–0.8581`；机制没有坍缩，也未退化为全图同号增强。

结论：P8 获得方向一致的 Dice/IoU、recall、Brier 和高频分层改善，第一次避免了 P6/P7 的小病灶 recall 损失；但 macro Dice 距预注册 `+0.002` 门槛少 `0.000152`，整体 precision 又下降 `0.001225`，且 Dice CI 跨 0，因此严格数值门为失败。当前不得访问 Test 或直接进入 150 epochs；应先基于 validation 做机制归因，并用独立预注册改动验证能否保留 recall/Brier 增益同时恢复 precision。


## RACE-PE V1：C4 → P9（2026-09-06，80 epochs）

用户选择 RACE-PE 后新增独立实现：区域 presence head 与像素 extent head 分开；presence 对齐文本存在性，extent 监督完整 mask，其区域均值对齐真实占比。文本位置缺失或含糊时使用 unknown；没有加入 boundary loss、LoRA 或 Lovasz。P8 的失败结论不变。

| ID | 完整 Git SHA | Git tag | 配置 |
|---|---|---|---|
| C4 | `add4908a0d6f702b0a10c4581725b535543829b8` | `pilot-c4-race-pe-control-80e-seed1219-20260906` | Frozen CXR-BERT + FAM-EPPA V4-B + Dice/Focal |
| P9 | `8129c1f039ed77f78e70305aca0bb9708b3b56b1` | `pilot-p9-race-pe-80e-seed1219-20260906` | C4 + RACE-PE；aux=0.05；bounded strength=0.15 |

- 开发提交：`8b31af87c6d1d7666d21100fb8f0183af790ecce`；设计说明：`D:/BetterLViT/race_pe_work/docs/RACE_PE_V1_DESIGN.md`。
- 两项统一：80 epochs、seed1219、physical batch16、deterministic、drop_last=True、阈值0.5。新配对按 validation macro IoU 选检查点，前5 epochs不参与选择；历史实验仍保留其原选择规则。
- 冻结源码的真实训练 batch16 前/反向分别重复两次，均通过且各自损失相同。四次初始输出 SHA-256 均为 `246feaa997468b4696ac8d02772f479d38f9315814ba220fed614be7abaa39b7`；C4/P9 allocated 峰值分别约15.10/17.33 GiB。单像素病灶、空mask、未知/否定文本、无效区域、辅助监督消融和IoU检查点选择行为检查均通过。
- 训练集文本/坐标审计：5716样本，4030个有保守可解析方位；1111个可解析明确单侧病例中1070个与当前图像坐标侧别一致。保留当前约定，不能把矩形六区描述为真实肺分割。
- 新服务器：`connect.westb.seetacloud.com:21465`，RTX4090D，PyTorch2.9.1+cu128。系统盘约22GiB可用，输出放在系统盘；不删除历史产物。
- 已发起后台 C4→P9 链，PID2757。状态路径：`/root/race_pe_runs/c4_p9_20260906/chain.status`；源码：`/root/BetterLViT-race-pe-c4`、`/root/BetterLViT-race-pe-p9`。已核对状态 `c4_training`，C4第1/80轮至少完成100/357个batch，GPU利用率99%，暂无训练错误；P9尚未开始，尚无新验证结果。
- `TEST_SPLIT_ALLOWED=0`、`AUTO_TEST_EVALUATE=0`。完成训练后只导出validation并生成`c4_vs_p9.json`，不自动扩展训练。
- 初筛门：macro IoU至少+0.003；整体Dice/precision、最小病灶四分位Dice/recall不下降，Brier不恶化。报告IoU配对bootstrap CI；即使单种子过门也不代表稳定增益。
- 已注册但未启动：C5（原RACE、同IoU选择规则）、C6（匹配权重的像素辅助监督、无路由）、C7（全部PE辅助监督、无路由）。后续先完成机制归因，再预注册至少3个配对种子；方法锁定前不访问Test。


### C4/P9 最终结果（2026-09-07核查）

链于悉尼时间2026-09-07 01:02:07完成，状态`complete`。C4/P9均完成80 epochs，两个validation导出及配对比较成功；日志未发现Traceback/RuntimeError/CUDA OOM，GPU空闲。两份JSON均为1429样本、阈值0.5、seed1219、IoU选检查点、test_split_accessed=false，提交来源与预注册一致。未启动额外训练或访问Test。

| 实验 | best epoch | Val macro Dice | IoU | Precision | Recall | Brier |
|---|---:|---:|---:|---:|---:|---:|
| C4 | 80 | 0.821214 | 0.722796 | 0.814701 | 0.864661 | 0.021477 |
| P9 | 69 | 0.822059 | 0.724335 | 0.796023 | 0.889474 | 0.021950 |

P9-C4：IoU +0.001538（95% CI [-0.001819, 0.004953]），Dice +0.000846；precision -0.018679（95% CI [-0.021869, -0.015604]），recall +0.024813；Brier +0.000473（恶化）。最小病灶四分位Dice -0.001677，IoU -0.000281，precision -0.035094，recall +0.047812。

结论：`passes_single_seed_screen=false`，IoU未达到+0.003且CI跨0；整体precision、最小病灶Dice和Brier亦未过门。不能称为稳定增益，不自动扩展至150 epochs或Test。

诊断线索：P9 best-checkpoint验证导出的最后一个batch中，文本槽概率均值0.998699；训练末轮均值0.9991。存在文本分支趋向全阳性的强烈信号；这些是末batch/末轮快照，不代表已经完成全验证集机制归因。V1使用positive-only文本监督，可能允许全阳性退化，需要单独诊断而非直接重启训练。

本地原始结果：`D:/BetterLViT/outputs/race_pe_results_20260907/`。归档SHA-256 `15421e4266f010d58115a9411f10ed849757d82e62d966c4097aa2b71a11f91f`已与服务器一致核验。


### C4/P9 用户授权 Test 检查（2026-09-07）

用户获知validation初筛失败后明确要求“用测试集看看结果”，本次授权覆盖C4/P9的Test评估。使用原validation选定的C4 epoch80与P9 epoch69，阈值固定0.5，未重训、未根据Test选检查点或阈值。历史validation JSON中的`test_split_accessed=false`描述当时状态；截至本次已访问Test。

两组完整评估2113张Test图像，导出与比较成功；来源提交仍为C4 `add4908a0d6f702b0a10c4581725b535543829b8`、P9 `8129c1f039ed77f78e70305aca0bb9708b3b56b1`。冻结训练工作树未改动；独立评估脚本沿用validation导出器的模型加载、预处理与指标实现，记录脚本SHA-256、授权和Test访问标记。

| 实验 | Test macro Dice | IoU | Precision | Recall | Brier |
|---|---:|---:|---:|---:|---:|
| C4 | 0.842989 | 0.756396 | 0.839363 | 0.874378 | 0.016938 |
| P9 | 0.841777 | 0.755351 | 0.820142 | 0.895348 | 0.017546 |

P9-C4：Dice -0.001212（95%配对图像bootstrap CI [-0.003527, 0.001090]）；IoU -0.001045（CI [-0.003706, 0.001705]）；precision -0.019221（CI [-0.021928, -0.016600]）；recall +0.020970（CI [0.017751, 0.024230]）；Brier +0.000608（恶化）。Test未显示IoU增益，IoU区间跨0不能声称显著下降；precision下降明确，仍不支持稳定增益。原validation失败结论不变，未启动后续训练。

原始Test结果、评估脚本、授权与日志：`D:/BetterLViT/outputs/race_pe_test_20260907/`。归档SHA-256 `85dec863d637fa0fb41e8c44546109e501caff10583f9cbbfb6075068642338e`已与服务器匹配；脚本哈希与2113样本数核验通过。


### RACE-PE退化诊断（2026-09-07，训练/验证专用）

已完成P9 epoch69的完整1429张验证图像前向干预、5716张训练图像标签/区域审计及16张验证图像的梯度探针。本次未重训或访问Test；此前用户授权的Test访问记录不变。原模型验证IoU精确复现0.7243345893510442。

- 确认文本全阳性退化：训练位置标签7894正例、0负例、26402未知。masked positive-only BCE允许全1解。完整验证8574个文本槽全部>0.99，均值0.998706。epoch1/20/69末验证batch槽均值约0.906/0.994/0.999。
- 仅将RACE槽设为全1或批内打乱，分别仅改变全部71701504验证像素中的20/16个二值预测，IoU均约0.724333。主LViT文本路径未改变，不能推断整个网络不使用文本。
- 关闭推理路由：IoU0.724149、precision0.802177、recall0.881141；相对native的precision回升0.006154，但仍比C4低0.012524。推理消融不等于独立训练的辅助监督消融，尚不能分离训练期间各分支的因果影响。
- 训练集无增强几何审计中，7894个报告派生正区域有1522个mask为空（19.28%）。固定图像矩形分区与肺部报告语义存在冲突；presence mask BCE要求0，文本positive consistency要求1。不能据此断言报告或mask错误。
- 验证四尺度视觉presence在真实空区域的误报率为99.02%/98.15%/88.06%/77.61%。Extent有一定前景/背景区分，但presence未有效校准。
- 4×4张验证图像的首层梯度探针中，辅助/主梯度范数比0.063–0.176，2/4组余弦为负；这是有限线索，不足以证明辅助梯度普遍主导训练。

后续优先修复文本目标可辨识性，将报告“提及”与疾病“存在”区分；移除或屏蔽与mask冲突的固定矩形正一致性。在小规模可学习性验证后再做匹配消融，不直接延长训练。未启动任何新训练。

完整报告与原始证据：`D:/BetterLViT/outputs/race_pe_diagnosis_20260907/诊断报告.md`。诊断归档SHA-256 `7e1544611e33c59142af43e9e1d92cdde983d156ce00f3dad4aa9849eb777dea`已与远端一致校验。


### RACE-PE V2最小修复与可学习性检查（2026-09-07）

V1的区域文本BCE只有正例、没有负例，允许全阳性解；报告区域与图像六个矩形区域也存在不一致。V2将监督定义为“编码器实际看见的文本明确提及哪些区域”，未提及作为这个语言任务的负例，不解释成疾病不存在；移除报告到视觉presence的正一致性损失。路由改为提及概率×视觉presence×像素extent。历史V1实现及C4/P9结果保持独立。

## 验证结果

- 历史V1行为检查和V2正/负提及、冲突空区域的梯度方向检查通过。
- 两次batch16完整CUDA前向/反向完全一致：loss 0.5188546180725098；输出SHA256 `246feaa997468b4696ac8d02772f479d38f9315814ba220fed614be7abaa39b7`；峰值分配17.33GiB，保留18.70GiB。
- 冻结CXR-BERT，仅训练随机初始化文本头，seed1219，固定40epoch，不按Val挑选epoch。
- Train共240种编码器可见模板，留出48种；拟合使用5200张图像的文本，Val1429张。

| 文本区域提及任务 | macro F1 | macro specificity | BCE |
|---|---:|---:|---:|
| Val正常文本 | 0.999180 | 0.996369 | 0.005928 |
| Train留出模板 | 0.984263 | 0.962965 | 0.101503 |
| Val循环打乱文本 | 0.615489 | 0.470361 | 3.322377 |
| Val全阳性预测 | 0.716333 | 0 | 4.233776 |

这些F1是规则派生的文本提及标签得分，并非分割Dice，也不是人工标注的临床语义准确率。常见报告模板在Train与Val间可以重复，因此额外按可见模板留出；留出分数按原始报告唯一项计算，同一可见模板可能有多个原始报告。该实验没有联合分割梯度，不能排除联合训练再次退化。

## 来源与边界

代码 `2f66ba898517dbee876ff6da735ac342943a08a2`，分支 `paper/race-pe-v2-development`，标签 `diagnostic-race-pe-v2-mentions-20260907`。本地与远端代码一致，远端仅有未跟踪的datasets软链接，指向 `/root/autodl-tmp/datasets`。运行环境见runtime.json。

原始归档SHA256 `3c55b851d1c7dc1c7c53311c78bbbdb2086d3fc77060344af7ffe0981445ec55`，已与远端核对。诊断文本头仅供复查，不自动用于后续分割初始化。

本次未访问Test，历史C4/P9已授权的Test访问事实不变。未启动新的完整分割训练，所有检查已结束，GPU空闲。

下一步应在Train/Val做匹配训练消融，分离“辅助监督”和“实际路由”的作用，再按预注册IoU/precision门槛决定是否扩大实验。固定矩形与肺解剖的错位仍是风险；文本头可学不等于路由可带来稳定增益。


### C8→P10 V2验证消融已启动（2026-09-07）

用户明确要求“启动”。悉尼时间2026-09-07 04:20:47启动后台链，先C8辅助监督，再P10辅助监督+路由，各80epoch、seed1219、batch16、Val IoU选择、阈值0.5、确定性训练。从头训练，不载入诊断文本头。复用已完成C4验证导出作无辅助监督基线。当前只是启动记录，不代表已完成或有增益。

- C8：`21606e02c2d64ae950c0f243f55163f58cbedf83`，tag `pilot-c8-race-pe-v2-80e-seed1219-20260907`。
- P10：`2f33a71219ed2beb339db45d702ecc36a931e0c8`，tag `pilot-p10-race-pe-v2-80e-seed1219-20260907`。
- 两组独立冻结manifest/工作树，仅路由开启与否不同；启动前batch16检查通过，初始loss0.5188546180725098和输出SHA完全一致。C8/P10峰值分配15.51/17.33GiB。
- 已实际观察C8第1轮60/357batch、GPU86%、显存17684MiB，无异常；P10等待自动接续。链PID29701，训练PID29710。
- 每组成功训练后自动导出完整Val，最终比较C4→C8、C4→P10、C8→P10。主门槛为P10对C4 macro IoU≥+0.003及既定非退化条件；路由归因还需C8→P10的正IoU区间且precision不退化。单种子不能证明稳定增益。
- 本轮`AUTO_TEST_EVALUATE=0`、`TEST_SPLIT_ALLOWED=0`，不会自动扩展或访问Test。此前C4/P9已访问Test记录保留。
- 远端输出：`/root/race_pe_v2_runs/c8_p10_20260907`。系统盘启动前可用20056121344字节，预计两组Best/Last加临时写入空间足够；不增加共享fs占用。

本地启动快照：`D:/BetterLViT/outputs/race_pe_v2_launch_20260907/launch_snapshot.json`。正式完成需核对chain.status、两个Val导出、三个比较JSON及来源提交。预注册说明见V2工作树`docs/RACE_PE_V2_SEGMENTATION_PILOT.md`。


### C8/P10 V2验证消融完成（2026-09-07）

悉尼时间14:51:06训练链complete，两组80轮及全部1429张Val导出、三组比较均完成。两组最佳检查点均为epoch80。远端冻结工作树跟踪文件干净；JSON来源与C8 `21606e02c2d64ae950c0f243f55163f58cbedf83`、P10 `2f33a71219ed2beb339db45d702ecc36a931e0c8`一致。GPU已空闲，未启动新实验，本轮未访问Test；此前C4/P9 Test访问记录不变。

| 实验 | Val macro IoU | Dice | Precision | Recall | Brier |
|---|---:|---:|---:|---:|---:|
| C4基线 | 0.722796 | 0.821214 | 0.814701 | 0.864661 | 0.021477 |
| C8仅辅助监督 | 0.726347 | 0.823727 | 0.818790 | 0.864762 | 0.021008 |
| P10辅助监督+路由 | 0.725594 | 0.823281 | 0.808676 | 0.876272 | 0.021305 |

C8-C4：IoU+0.003551，配对图像bootstrap95%CI[0.001131,0.006045]；precision+0.004088；最小病灶四分位Dice+0.002404、recall+0.002051；Brier降低。通过预注册单种子筛选。但这是同一seed的图像抽样区间，不证明跨训练种子的稳定增益。

P10-C4：IoU+0.002798，CI[-0.000625,0.006320]，未达+0.003门槛；precision-0.006025；最小病灶Dice-0.000826，筛选失败。P10-C8：IoU-0.000753，CI[-0.003779,0.002389]，不能声称显著IoU下降；precision-0.010113，CI[-0.013085,-0.007154]；recall+0.011510；Brier+0.000297。因此未证明路由带来增益，仍有扩大预测范围的表现。优先进一步核验C8辅助监督路径，而非延长P10或宣称完整RACE-PE创新成功。

结果目录 `D:/BetterLViT/outputs/race_pe_v2_results_20260907/`；下载归档SHA256 `4baa2e639eb43e4d966542fa8254762cf910af85e64824e571b79b147887760a`已与远端一致核验。后续需诊断辅助监督贡献并预注册额外配对种子；本次没有自动扩展。


### C8/P10用户授权Test结果（2026-09-07）

用户明确要求“结果以测试集为准”。最终性能汇报以Test macro Dice/IoU为主，Val用于训练选择和机制筛选。此次授权评估C8/P10的原Val IoU选定epoch80检查点，固定阈值0.5，无重训、Test挑检查点或Test调阈值。复用C4历史同协议Test结果；各2113样本，比较按图像配对。原冻结训练manifest的Test禁止标记描述训练阶段，本次独立评估已获得后续用户授权。截至本次C8/P10也已访问Test，不能再称其Test未访问。

| 实验 | Test macro Dice | IoU | Precision | Recall | Brier |
|---|---:|---:|---:|---:|---:|
| C4基线 | 0.842989 | 0.756396 | 0.839363 | 0.874378 | 0.016938 |
| C8仅辅助监督 | 0.844313 | 0.758612 | 0.843207 | 0.873427 | 0.016740 |
| P10辅助监督+路由 | 0.843951 | 0.758090 | 0.832287 | 0.885560 | 0.016846 |

C8-C4：Dice+0.001324，配对图像bootstrap95%CI[-0.000605,0.003290]；IoU+0.002216，CI[-0.000004467,0.004463]；precision+0.003844，CI[0.001823,0.005844]。Test数值有小幅改善，但Dice和IoU区间均跨0，不能声称显著或稳定增益；Val单种子通过仍只是Val结论。

P10-C4：Dice+0.000962，CI[-0.001635,0.003569]；IoU+0.001694，CI[-0.001309,0.004688]；precision-0.007076，CI[-0.009901,-0.004268]。P10-C8：Dice-0.000362、IoU-0.000521，两者区间跨0；precision-0.010921，CI[-0.013471,-0.008407]。完整路由仍未证明增益。额外配对种子与机制归因尚未完成。

两组来源分别为C8 `21606e02c2d64ae950c0f243f55163f58cbedf83`、P10 `2f33a71219ed2beb339db45d702ecc36a931e0c8`；脚本哈希、来源SHA、最佳epoch和样本数均核验。结果目录 `D:/BetterLViT/outputs/race_pe_v2_test_20260907/`，归档SHA256 `01f31f2052ddf4b6eaa6d9dedd1c3183e439ed996783782e11c182cf90e6b9de`与远端一致。status complete，无新训练。

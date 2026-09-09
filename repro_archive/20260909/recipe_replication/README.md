# R2的额外配对种子复验

## R2-2027首次检查

2026-09-09 12:56:37.379（悉尼）的第1次短连接确认R2-2027训练健康：完成4/80轮、正在第5轮，SHA/manifest一致、跟踪文件干净，未访问Test。排除第1轮后，第2–4轮平均214.928532秒，预计今天17:27:38.861结束训练；同一heartbeat `betterlvit-r2` 已改约17:40进行第2次也是最后一次检查。预算1/2，之后不再中途查询或保持连接。预测不保证实际间隔，末检须按真实结束时间核对≤30分钟。首检快照、预测、state和定时记录均已归档；当前没有R2-2027最终结果，不根据早期指标判断配对成败。

## R2-2027已提交

C4-2027完整结果归档并推送后，R2-2027于2026-09-09 12:40:37.904（悉尼）提交，PID149524，来源 `c972a152b5d152ddfa82e034596085aca85dd0ab`，tag `experiment-r2-recipe-80e-seed2027-20260909`。当前只有启动回执，尚未检查健康，独立预算0/2。同一heartbeat `betterlvit-r2` 已改约今天12:56首次检查，再按实测整轮时间预测末检。启动前系统盘可用9,163,235,328字节，原模型均保留。配方完全沿用预注册设置；完成后只与C4-2027作同种子比较。两组3407来源已准备，尚未提交。

## C4-2027完成

已完成80轮，Best70，Val macro IoU **0.7241900931**、Dice **0.8228644537**，1429样本、固定阈值0.5，未访问Test。训练12:27:33（悉尼）结束，Val导出12:29:31完成，末检12:37:40，结束至末检607.068秒；预算2/2，符合30分钟约定。完整日志、实际80轮LR、逐图记录、Best/Last来源及源文件哈希已核对，见[c4s2027_results](c4s2027_results/README.md)。下一步按原计划提交匹配的R2-2027；不能用跨种子数值代替配对比较。

## 首组首次检查

2026-09-09 07:54:39（悉尼）的第1次短连接确认C4-2027训练健康：完成4/80轮、正在第5轮，SHA/manifest一致，跟踪文件干净，未访问Test。排除第1轮后，第2–4轮平均214.800541秒，预计今天12:24:59.974结束训练；同一heartbeat `betterlvit-r2` 已改约12:37进行第2次也是最后一次检查。预算1/2，之后不再中途查询或保持连接。预测不保证实际间隔，末检须根据真实结束时间验证≤30分钟。首检快照、预测、state和定时更新记录均已归档；当前没有本种子最终Val/Test结果。

## 首组已提交

2026-09-09 07:38:08（悉尼）提交C4-2027，PID138765，来源 `965382b24ad26de81b89f9ddfe9ab434ae4d105b`，tag `experiment-c4-recipe-80e-seed2027-20260909`。当前仅有启动回执，尚未确认正式训练健康，独立检查预算0/2。新heartbeat `betterlvit-r2` 已预约今天07:54首次短连接；此后根据实测整轮时间改约最终检查。旧筛选heartbeat `betterlvit-r1-r2` 已删除。其余三组准备完成，尚未提交。启动前系统盘可用10,876,534,784字节。

完整启动/state/chain及预约记录均保存在本目录，实时状态仍以outputs目录为准。

这是用户已授权计划的下一阶段：seed1219的R2通过Val筛选，补齐2027、3407两个匹配种子。完整预注册说明见 `docs/RECIPE_REPLICATION_PLAN_20260909.md`（代码和文档分支均保存副本）。R1不扩展、不组合；当前不访问Test。

| 顺序/label | profile | seed | 比较角色 |
|---|---|---:|---|
| c4s2027 | c4_race_pe_control | 2027 | 同种子控制 |
| r2s2027 | r2_single_cosine | 2027 | 单次余弦候选 |
| c4s3407 | c4_race_pe_control | 3407 | 同种子控制 |
| r2s3407 | r2_single_cosine | 3407 | 单次余弦候选 |

每个训练80轮、batch16、224、原增强、原Dice/Focal、Adam，seed只在同种子配对内相同。各组独立完整SHA、tag、manifest及工作树，详见 `sources.json`。中间数值不改变队列，不因2027结果不好就丢弃3407，也不追加选择性种子。

训练源自已验证代码，仅扩展启动manifest允许C4和已注册种子、预检记录seed；模型、数据、主训练循环、loss和学习率数学与原版本逐文件一致。每个来源两次真实Train批次GPU预检，五步临时权重不用于正式训练；`verify_preflights.py`要求同一seed的C4/R2初始模型、增强输入、五步输出和loss全部一致。

部署时系统盘可用10,881,875,968字节，四组准备完成后10,877,665,280字节；共享fs为18,559,782,256字节（<20,000,000,000）。四组预计新增约6.8GB的Best/Last，每次启动仍检查至少4GB可用。保留全部原始模型。HF认证问题在前阶段已告知，未收到认证变化时不要再次重试/通知；已完成模型可先核对来源、生成Xet清单并保留。

## 全部预检已通过

四个冻结来源各两次真实Train批次五步预检均通过；两个种子内的C4/R2初始模型、输入、五步输出和loss全部相同，不同种子的初始权重不同。完整proof见 `preflight_verification.json`，每个来源的原始预检输出单独保存。尚未把任何临时预检权重用于正式训练。

## 实时路径与执行

- 当前脚本：`D:/BetterLViT/experiment_docs_work/repro_archive/20260909/recipe_replication`。
- 可变状态：`D:/BetterLViT/outputs/recipe_replication_20260909`，`chain_state.json`指明当前label；每个`<label>_state.json`独立计检查次数。
- 服务器训练：`/root/recipe_runs/<label>_80_20260909`；工作树见sources.json。
- Git归档的state只是历史副本，不能替代outputs中的预算；不改正在训练的源码，不合并主分支。

1. `launch_replication.py --label <label>`校验前一训练已完成、全部配对预检通过、空间及SHA，然后只发起后台进程并返回PID。写出launch/state和当前chain_state，预算0/2；不得把回执当作训练健康证明。
2. 约15分钟后的首次预约，用 `collect_snapshot.py --state <outputs>/<label>_state.json --output <outputs>/<label>_first_snapshot.json` 做唯一的一次短连接。健康且至少完成3轮时，运行 `forecast_recipe.py --label <label>`，把同一heartbeat改约 `planned_final_check_sydney`。不继续查服务器，正常运行不通知。
3. 最后预约以同一collector保存 `<label>_final_snapshot.json`，然后本地运行 `finalize_replication.py --label <label>`。核实80轮、1429样本、种子与来源、真实结束到检查的间隔。收集器连接前扣预算，禁止第三次查询和隐藏重试。若尚未结束或故障，保留事实并处理，不能继续下一组占用GPU。
4. 已确认完成后运行 `archive_completed_recipe.py --label <label>`，只下载已完成静态文件，核对哈希及80个实际LR。可运行 `hf_recipe.py prepare --label <label>`核对Best/Last及生成Xet源清单，不进行无条件认证重试。其远端准备脚本沿用前阶段已归档的`prepare_hf_recipe_server.py`。已完成文件传输不作为新增训练状态检查。
5. C4完成时只归档对照；同种子R2完成时会自动生成 `c4_vs_r2_seed<seed>.json`。单个配对的门槛失败不能阻止另一已注册种子的执行。更新文档/台账并推送GitHub后，按sources的previous_label顺序提交下一组，预约新的首次检查，独立预算0/2。
6. 四组完成后运行 `aggregate_replications.py`，结合已完成seed1219，汇报三个配对IoU/Dice差值、均值和样本标准差。三个IoU差值均>0、平均≥0.003、平均Dice及最小面积组IoU不降才通过复验；逐图bootstrap不是跨种子显著性证据。
7. 复验通过则在固定方法、checkpoint与阈值规则下完成后续Test评估，正式结果以Test为准；失败则报告并结束这条配方支线，不扩150轮、不追加种子或改变配置。每个阶段源码和原始结果均提交推送。当前定时目的结束后删除对应heartbeat，避免无效提醒。

初始化脚本`register_replication.py`/`prepare_replication.py deploy`只用于一次性复现准备，已有目录时不能盲目重跑。常规接续只用已经冻结、预检完成的四组来源。

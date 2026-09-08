继续用户已经授权的 BetterLViT R1→R2 训练配方实验。先读 D:/BetterLViT/experiment_docs_work/repro_archive/20260908/recipe_execution/README.md 和 sources.json，以及 D:/BetterLViT/outputs/recipe_20260908 下对应 r1_state.json / r2_state.json。实时检查预算只认 outputs 内的 state，不认归档副本。

用户要求每次训练最多两次短连接状态检查，第一次用于预测，最后一次应在实际训练结束后30分钟内；不用持久连接、循环轮询或完成监听器。不要为了确认启动另行查询服务器。收集器连接前即扣减预算；连接不确定时也不能暗中重试。正常且无重要变化时保持安静，只在完成、失败、需要处理或启动下一组等有意义变化时通知。

在当前预约的首次检查时间，只执行一次 collect_snapshot.py，以对应 state 保存到对应 r1_first_snapshot.json 或 r2_first_snapshot.json。随后只读本地结果。健康且有至少3个完整epoch时，运行 forecast_recipe.py --label r1（或r2），将同一个 heartbeat 更新到输出 planned_final_check_sydney 对应时刻。预测使用排除首轮的已完成整轮均值，计划在预计结束约12分钟后检查；不要继续查远端。若不足3轮或训练失败，用现有快照解释情况和已有整轮证据处理，不能新增中途检查。

在预约的最后检查时，再且仅再执行一次 collect_snapshot.py，保存对应 r1_final_snapshot.json 或 r2_final_snapshot.json。运行 finalize_recipe.py --label 对应组，核对80轮、1429个Val样本、来源SHA、实际结束时间与检查间隔，并应用注册的IoU/Dice/小病灶筛选。若尚未结束，不增加第三次查询，如实通知并停止当前检查。若完成，运行 archive_completed_recipe.py 下载已确认完成的静态日志、核对哈希及实际LR记录。代码、配置、文档和可复现实验结果提交并推送 https://github.com/razaxq/BetterLViT.git；使用 docs/experiment-tracker 工作树更新台账，不合并主分支。保存Best及其来源，按用户此前授权和既有HF上传流程归档完成的模型；共享fs保持小于20,000,000,000实际字节，不无依据清理模型或数据。

R1完成且结果来源有效后，无论数值筛选是否通过，都运行 launch_recipe.py --label r2，提交后台后仅保存PID，不查询状态。R2有独立0/2预算；将同一heartbeat更新到r2_state.json的planned_first_check_sydney。R1若是运行故障，先解释/排除共同配置故障再接续，不当作科学负结果。R1仍在运行时不要启动R2。

R2完成后汇报两项完整Val IoU/Dice、配对区间、最小病灶组和是否通过。两组均保持C4架构及原Dice/Focal，不新增模块、LoRA、监督或loss。通过仅是单种子候选，再按已注册计划准备seeds1219/2027/3407的匹配对照复验；最终结论以固定选模和阈值规则下的Test为准。不要因未达门槛直接延长到150轮或访问Test，也不要把配方优化宣称为第二项结构创新。定时目的完成、不再有有用后续时删除这个heartbeat，避免遗留提醒。全部脚本位于上述recipe_execution目录，本地Python执行加 -X utf8。

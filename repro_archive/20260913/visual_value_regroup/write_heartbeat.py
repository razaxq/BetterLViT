"""Prepare native heartbeat settings from saved run state; never edit app TOML."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from remote_ops import HERE,read,save
p=argparse.ArgumentParser();p.add_argument('--label',choices=('m1','m2'),required=True)
p.add_argument('--phase',choices=('first','final'),required=True);a=p.parse_args()
state=read(HERE/(a.label+'_state.json'))
key='planned_first_check_unix' if a.phase=='first' else 'final_check_unix'
at=datetime.fromtimestamp(state[key],ZoneInfo('Australia/Sydney'))
prompt=f'''继续用户已授权的第二创新点发现实验，工作目录D:/BetterLViT/experiment_docs_work，当前唯一执行目录repro_archive/20260913/visual_value_regroup。先读README.md、PROTOCOL.md、sources.json、preflight_verified.json和{a.label}_state.json。当前对象{a.label}，预约{a.phase}检查，悉尼{at.isoformat()}。本批M1视觉锚点聚合与M2文字锚点聚合均16896参数，新增值分支只读取图像；各80轮seed1219，冻结CXR-BERT、无LoRA、Dice/Focal、boundary0、EPPA V4-B、原R2配方。基线R2复用9eca26de5b301099805530edbf5a1a8718bea662。当前仍是候选，不能声称已找到第二创新点。旧T1/T2和T2干预已结束，不再检查或启动。

每组最多两次训练状态检查，不保持SSH连接。先读本地state判定是否到期、是否重复、检查次数；未到期或重复触发则不访问服务器。到期只调用一次 python -X utf8 repro_archive/20260913/visual_value_regroup/inspect_run.py --label {a.label} --phase {a.phase}。首检若training，则运行verify_first_snapshot.py --label {a.label}，用write_heartbeat.py --label {a.label} --phase final生成新设置，并通过automation_update更新同一个原生heartbeat，随后verify_schedule.py核验。预测80轮结束后12分钟检查，实际目标<=30分钟。不得增加第三次常规轮询。若失败或最终预约过早，保留异常事实再处理，不重复训练、不改检查预算。健康且仍运行的首检保持安静。

仅当保存snapshot显示complete后，依序执行同目录archive_completed.py --label {a.label}、analyze.py --label {a.label}、upload_completed.py --label {a.label}、verify_downloads.py --label {a.label}。读取已完成静态证据不算训练状态轮询。必须检查80轮/Best/source、实际LR、完整1429张Val、固定>0.5逐图macro、四个Train观察、训练/验证退出码及日志；不能在Val运行时报告分数。若export与checkpoint选择IoU差>=1e-7，先核验并解释>=与>阈值差异，不改指标或门槛。更新台账后publish.py --message 简洁完整描述 --completed-label {a.label}。所有权重、源码和证据按训练完整SHA对应短前缀加性备份HuggingFace并双端size/Xet核验；不要输出凭据。GitHub实际远程是https://github.com/razaxq/BetterLViT.git，origin为本地bare，不要改历史训练工作树。

M1完整完成、备份、下载核验和GitHub发布之后，无论M1排名好坏均执行launch.py --label m2，然后write_heartbeat.py --label m2 --phase first生成设置并更新同一个heartbeat、verify_schedule.py核验。不要重复启动。M2已发出则不得再启动M1。每次启动需scratch>4GB、shared实际<20000000000字节。四个指定历史Last的清理在本批启动前已完成且有备份证据，不能重复执行；保留所有Best、近期T1/T2及RS1 Best/Last、F/B模型缓存、数据和环境。

M2完成并完整归档后，运行assess_discovery.py，按事前门槛评判：M2-R2 IoU>=0.003且分组描述性CI下限>0，M2-M1 IoU>0且CI下限>0，Dice及最小病灶四分位IoU对两组都不退化。报告IoU/Dice、precision/recall、Brier、小病灶和FP/FN。只属于单种子Val发现证据，无新Test访问。通过后才准备配对额外种子及机制消融；不通过则关闭本版本并保留负结果，不降低门槛或改名认领创新。OCR/RecLMIS已有类似分组/重建思想，任何创新结论还需要文献差异化和机制证据。第二创新点目标仍需后续验证。

两组结束后提交最终报告、判据JSON和台账，推送GitHub，删除本批heartbeat，避免再次触发旧实验。所有自动化只能通过原生工具创建/更新/删除，不直接编辑TOML。'''
if a.label=='m2':
    proof=read(HERE/'m1_results/independent_verification.json')
    backup=read(HERE/'m1_results/hf_upload_verified.json')
    assert proof['verified'] and backup['verified'] and backup['independent_local_listing_verified']
    prompt+='''

M2阶段限定：M1已完成80轮、Best67、完整Val及云端备份、下载核验和GitHub归档，M2也已派发。上文M1完成后启动M2的步骤已经执行，禁止再次调用launch.py --label m2，也不要重复归档或检查M1。当前仅处理M2的本次预约及后续唯一末检。M1 Val IoU72.7827%、Dice82.5333%，相对R2 IoU+0.1173pp，分组描述性95%CI[-0.1469,+0.3871]pp，最小病灶IoU-0.0941pp；不构成稳定收益证据。M2结束后与原R2及此M1同时比较，并按预注册门槛作发现阶段决策。'''
(HERE/'heartbeat_prompt.txt').write_text(prompt+'\n',encoding='utf-8',newline='\n')
settings=dict(mode='create',kind='heartbeat',destination='thread',name='BetterLViT M1 M2',prompt=prompt,status='ACTIVE',
    rrule=f'FREQ=DAILY;BYHOUR={at.hour};BYMINUTE={at.minute};BYSECOND=0')
identity=HERE/'automation_identity.json'
if identity.exists():
    ident=read(identity)
    settings.update(mode='update',id=ident['id'],targetThreadId=ident['target_thread_id'])
save('automation_update_request.json',settings)
print(at.isoformat())

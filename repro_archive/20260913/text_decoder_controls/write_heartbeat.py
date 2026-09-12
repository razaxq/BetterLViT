"""Prepare user-visible native heartbeat settings; does not edit app TOML."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from remote_ops import HERE,read,save
p=argparse.ArgumentParser();p.add_argument('--label',choices=('t1','t2'),required=True)
p.add_argument('--phase',choices=('first','final'),required=True);a=p.parse_args()
s=read(HERE/(a.label+'_state.json'))
key='planned_first_check_unix' if a.phase=='first' else 'final_check_unix'
local=datetime.fromtimestamp(s[key],ZoneInfo('Australia/Sydney'))
next_step='完成T1后，无论点估计好坏均先验证备份和GitHub发布，再调用launch.py --label t2；独立冻结的T2训练不能按T1早期排名取消。随后write_heartbeat.py --label t2 --phase first生成新原生设置，使用automation_update更新现有betterlvit，并verify_schedule.py核验。'
completed_context=''
if a.label=='t2':
    next_step='T1已完整完成、备份和发布，T2已经提交；不得重复检查T1或重复启动任何一组。T2首检仅更新本组唯一末检预约。'
    proof=read(HERE/'t1_results/independent_verification.json')
    pair=read(HERE/'t1_results/paired_comparison.json')
    backup=read(HERE/'t1_results/hf_upload_verified.json')
    assert proof['verified'] and backup['verified'] and backup['independent_local_listing_verified']
    completed_context=(f"\n\nT1完成背景：Best{proof['best_epoch']}，Val IoU{pair['candidate']['iou']*100:.4f}%、Dice{pair['candidate']['dice']*100:.4f}%；"
        f"相对R2 IoU{pair['deltas']['iou']['mean']*100:+.4f} pp，分组描述性区间{[round(x*100,4) for x in pair['deltas']['iou']['ci95']]} pp，"
        f"small-mask IoU{pair['deltas']['iou']['small_mean']*100:+.4f} pp。无稳定收益证据，也未访问Test。检查2/2、末检距训练结束{proof['seconds_after_training_end']:.3f}秒。"
        f"HF {backup['bucket']}/{backup['bucket_prefix']}/共{backup['verified_files']}文件、{backup['logical_bytes']}字节已双端核验，Best/Last保留。详见t1_results/REPORT.md、paired_comparison.json及备份回执。")
prompt=f'''继续用户已授权的BetterLViT文字IoU实验。工作目录D:/BetterLViT/experiment_docs_work；唯一当前执行档案repro_archive/20260913/text_decoder_controls。先读README.md、PROTOCOL.md、sources.json、preflight_verified.json、{a.label}_state.json及台账。当前对象{a.label}，本次是{a.phase}检查，约定悉尼{local.isoformat()}。不得启动或轮询旧RS1及其他旧实验。T1来源72295aa38649fea8ffed2cdd7330c3a18504a330，T2来源488ef093de80df71ee77741a8c7ee7b938c7d6b5；各80轮seed1219、原R2配方、16896参数、noLoRA、Dice/Focal、boundary=0、legacy增强、单次余弦。T0复用原R2 9eca26de5b301099805530edbf5a1a8718bea662，关闭新分支的五步grouped Adam与原R2已精确一致。当前是机制筛选，仅Val；不宣称Test增益或第二创新点成立。

每个正式训练最多两次预测检查，不保持SSH训练连接，不额外查询日志/GPU。先用本地state判断次数和计划时间：未到期、重复触发或次数耗尽则不访问服务器。到期只调用一次 python -X utf8 repro_archive/20260913/text_decoder_controls/inspect_run.py --label {a.label} --phase {a.phase}。首检用完整post-warmup轮耗时预测80轮终点，设置原生heartbeat betterlvit到final_check_unix（预测终点后12分钟）并核验实际TOML；不要建立重复监控。正常进行中保持安静；完成、失败或需用户处理才通知。末检需记录实际距训练结束时间，目标<=1800秒，不能用第三次常规检查掩盖超预算；异常保留事实并诊断。

仅当snapshot显示complete后，执行同目录archive_completed.py --label {a.label}、analyze.py --label {a.label}、upload_completed.py --label {a.label}、verify_downloads.py --label {a.label}、publish.py --message 合适的完整描述 --completed-label {a.label}。这些读取已完成静态产物不另算训练轮询。自动Val已由冻结runner串联；必须等待完成JSON，不能在评估运行时报告成绩。核查80轮/Best/source、整数计数macro、固定阈值、四个Train观察及日志；若export与选择IoU差>=1e-7则先解释>=与>阈值差异，必要时对完成Best做只读Val核验，不能掩盖。报告IoU/Dice、small-mask、paired grouped CI；T2完成后还需t2_vs_t1.json。

{next_step}T2完成后总结三组，说明单种子发现性质，暂停此监控。后续短语绑定/图像证据控制根据结果继续研究，未通过文字对照不直接声称创新。完整负结果也必须提交。

保留当前RS1 Best/Last、F/B缓存、数据和环境。五个老Last已在新历史备份补齐且独立hash/引用检查后清理；不可重复清理。每次新启动仍需scratch>4GB、shared实际<20000000000字节。HF用缓存凭据、AutoDL学术加速，按源短SHA加性上传并双端size/Xet验证，不输出token，不删除云端。所有新增源代码、文档和回执提交并推送GitHub；origin是本地bare，实际推送https://github.com/razaxq/BetterLViT.git。不要改历史训练工作树。'''
prompt+=completed_context
(HERE/'heartbeat_prompt.txt').write_text(prompt+'\n',encoding='utf-8',newline='\n')
settings=dict(mode='update',id='betterlvit',kind='heartbeat',name='BetterLViT 区域监督',
    prompt=prompt,status='ACTIVE',rrule=f'FREQ=DAILY;BYHOUR={local.hour};BYMINUTE={local.minute};BYSECOND=0',
    targetThreadId='01a074f2-5c58-7221-9a58-59850e626880')
save('automation_update_request.json',settings)
print(f'{a.label} {a.phase}: {local.isoformat()}')

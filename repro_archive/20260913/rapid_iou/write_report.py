"""Update the experiment tracker with completed evidence and explicit pending work."""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
DOCS=HERE.parents[2]
def read(p):return json.loads(Path(p).read_text())
plan=read(HERE/'plan.json')
lines=['# R2单模型：快速IoU工程验证','',
    '目标是复用现有R2取得实际IoU提升。当前仅一个seed1219、Best67模型，不增加参数、模型推理次数或训练。'
    '不执行集成、TTA或小区域过滤。阈值优化不作为第二创新点。','',
    '模型来源 `'+plan['sources'][0]['source_git_commit']+'`；原0.5成绩保持不变。'
    '阈值只在1429张Val的固定41点网格上选择，通过预登记门后冻结，再评估2113张Test。']
for split in ('validation','test'):
    result_path=HERE/split/'summary.json'
    launch_path=HERE/(split+'_launch.json')
    if result_path.exists():
        s=read(result_path);assert s['verified']
        lines+=['',f'## {split} 已完成','',
            '| 设置 | 阈值 | macro IoU | macro Dice | Precision | Recall |',
            '|---|---:|---:|---:|---:|---:|']
        for name,k,t in [('原R2','baseline',.5),('Val选定阈值','candidate',s['threshold'])]:
            lines.append(f'| {name} | {t:.2f} | '+' | '.join(f'{s[k][m]*100:.4f}%' for m in ('iou','dice','precision','recall'))+' |')
        ci=s['bootstrap']['ci95']
        lines+=['',f"IoU差值{s['delta']['iou']*100:+.4f} pp，分组描述性95%区间[{ci[0]*100:+.4f},{ci[1]*100:+.4f}] pp。"
            f"最小面积{s['small_count']}张IoU差值{s['small_delta']['iou']*100:+.4f} pp；"
            f"Dice差值{s['small_delta']['dice']*100:+.4f} pp，Recall差值{s['small_delta']['recall']*100:+.4f} pp。"]
        if split=='validation':
            lines+=['',f"Val推进门通过={s['passed_validation_gate']}。Val用于选阈值，本身存在选择乐观偏差。"]
        else:
            lines+=['','阈值在Test前冻结并提交GitHub；Test未搜索阈值。所有原0.5逐图TP/预测面积/真值面积与历史结果精确一致，'
                '新结果已独立从整数计数复算。该Test历史上已经被访问；本次为单一训练种子的工程评估，不是跨种子稳定性或新结构证据。']
    elif launch_path.exists():
        launch=read(launch_path)
        when=datetime.fromtimestamp(launch['started_unix'],ZoneInfo('Australia/Sydney')).isoformat()
        lines+=['',f'{split} 已于{when}后台提交；尚无核验完成结果。']
selection=HERE/'selection.json'
if selection.exists():
    s=read(selection)
    if not s['methods']:
        lines+=['','本轮Val未通过预登记推进门，未启动新Test，继续保留R2原阈值0.5。']
    elif (HERE/'test/summary.json').exists():
        t=read(HERE/'test/summary.json')
        lines+=['',f"本轮已结束：Test IoU点估计变化{t['delta']['iou']*100:+.4f} pp。"
            '参数、权重、图像/文字输入及前向次数完全保持；结果归因于Val选定阈值。']
    else:
        lines+=['','Val阈值已冻结，Test最终结论仍待完成。']
lines+=['','完整协议、冻结计划、源码、部署和下载哈希、逐图整数计数及阈值选择回执均保存在本目录。'
    '没有新训练或新checkpoint，因此不复制已有的大型模型文件。']
(HERE/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
tracker=DOCS/'docs/EXPERIMENT_TRACKER.md';text=tracker.read_text(encoding='utf-8')
heading='## R2单模型快速IoU验证（2026-09-13）'
section=heading+'\n\n'+'\n'.join(line.replace('## validation','### validation').replace('## test','### test') for line in lines[2:])
section+='\n\n[完整执行与结果](../repro_archive/20260913/rapid_iou/README.md)。\n\n'
if heading in text:
    start=text.index(heading);end=text.index('\n## ',start+len(heading));text=text[:start]+section+text[end+1:]
else:
    anchor='## 中间解码层文字对照 T1/T2（2026-09-13）'
    assert anchor in text;text=text.replace(anchor,section+anchor,1)
tracker.write_text(text,encoding='utf-8',newline='\n')

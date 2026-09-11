"""Generate complete, nonselective crossfit report only after independent checks."""
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo
HERE=Path(__file__).resolve().parent
R=HERE/'results';s=json.loads((R/'summary_verified.json').read_text())
assert json.loads((R/'independent_verification.json').read_text())['verified']
assert json.loads((HERE/'checkpoint_verified.json').read_text())['verified']
runtime=json.loads((R/'runtime.json').read_text());state=json.loads((HERE/'state.json').read_text())
finished=datetime.fromtimestamp(runtime['completed_unix'],ZoneInfo('Australia/Sydney')).isoformat()
text=f'''# F：局部纠错接口的五折筛选结果

完成时间{finished}。源`{runtime['source_git_commit']}`，5折×4头×2048步，全部4585张原B fit各有一个新增头OOF预测。原R2曾训练全部图像，以下仅是新增头的Train内交叉拟合开发结果，不是官方Val或Test。

完整SHA/缓存身份/计数/转移/训练预算与同组不跨折核验通过，20个头CPU weights_only严格载入、状态哈希和张量有限性核验通过。各折只在最后一步评自身留出一次，不选择中间checkpoint。旧B holdout、官方Val/Test未访问。

| 接口 | macro IoU % | macro Dice % | ΔIoU pp | group CI95 pp | 正向折数 | 基础门 |
|---|---:|---:|---:|---|---:|---|
'''
text+=f"| R2固定输出 | {s['means']['baseline']['iou']*100:.6f} | {s['means']['baseline']['dice']*100:.6f} | — | — | — | — |\n"
for v in ('coarse_free','coarse_mass','fine_free','fine_mass'):
    c=s['comparisons'][v+'-baseline'];m=s['means'][v];g=s['baseline_gates'][v]
    text+=f"| {v} | {m['iou']*100:.6f} | {m['dice']*100:.6f} | {c['delta_iou']*100:+.4f} | [{c['group_ci95'][0]*100:+.4f}, {c['group_ci95'][1]*100:+.4f}] | {sum(x>0 for x in c['fold_delta_iou'].values())}/5 | {'通过' if g['passed'] else '失败'} |\n"
text+='\n## 全部门与误差转移\n\n'
for v in ('coarse_free','coarse_mass','fine_free','fine_mass'):
    failed=[k for k,ok in s['baseline_gates'][v]['checks'].items() if not ok];t=s['transitions'][v]
    text+=f"- **{v}**：未通过项{failed or '无'}；每图修好FP/FN为{t['fixed_fp']:.3f}/{t['fixed_fn']:.3f}，新增FP/FN为{t['new_fp']:.3f}/{t['new_fn']:.3f}。\n"
text+='\n## 各折相对R2的macro IoU增量（百分点）\n\n| 折 | 样本数 | coarse_free | coarse_mass | fine_free | fine_mass |\n|---|---:|---:|---:|---:|---:|\n'
counts=json.loads((HERE/'manifest.json').read_text())['fold_counts']
for f in range(5):
    values=[s['comparisons'][v+'-baseline']['fold_delta_iou'][str(f)]*100 for v in ('coarse_free','coarse_mass','fine_free','fine_mass')]
    text+=f"| {f} | {counts[str(f)]} | "+' | '.join(f'{x:+.4f}' for x in values)+' |\n'
text+='\n## 细接口的额外作用\n\n'
for v,c in (('fine_free','coarse_free'),('fine_mass','coarse_mass')):
    p=s['comparisons'][v+'-'+c]
    text+=f"- {v}−{c}：ΔIoU {p['delta_iou']*100:+.4f} pp，group CI [{p['group_ci95'][0]*100:+.4f}, {p['group_ci95'][1]*100:+.4f}] pp；细接口门：{s['fine_interface_gates'][v]}。\n"
text+=f'''
预注册规则给出的后续文字对照接口：**{s['recommended_interface_for_text_screen'] or '无候选通过，停止直接推进'}**。该判断不授权改变本轮配置、追加训练、重新调旧折或访问Test。细接口读取的视觉细节与空间耦合同时改变，不把收益全归因于单一分辨率或新增文字；本轮四头没有额外文本输入，原R2仍包含文字。

检查次数{state['inspections_completed']}/2，距全部完成{state.get('completed_unix_inspection_delay_seconds',float('nan'))/60:.3f}分钟。全部样本、完整五折与失败项已报告；只有通过接口门后才按独立协议研究文字增量。当前并不证明稳定Test IoU或第二创新点。
'''
text+='''
## 本轮失败后的诊断与处理

四个分支都在学习，平均候选残差幅度约0.128–0.241，每图改变约96–119个像素；本轮没有复现B/D的零输出塌缩。

无保量两组仍偏向扩张前景：coarse_free每图净增FP约48.445，只减少FN约25.583；fine_free净增FP约42.951、减少FN约23.604。细尺度无保量相对粗版本存在小幅正向差异，但两者均低于固定R2，未通过基础门。

保量限制了这种扩张，两组在五折都出现小幅正向开发信号。fine_mass相对R2为+0.05435 pp、group CI[+0.03741,+0.07164] pp；但不足预注册+0.1 pp，Precision下降0.03156 pp。每图约49.49个原错被修好、48.62个原正确像素被破坏，净纠错余量仍很小。不能把95%区间正向等同于效应足够大或Test稳定。

fine_mass相对coarse_mass仅+0.00099 pp，group CI[-0.00624,+0.00829] pp，无法证明保量后读入细尺度特征有额外收益。因此E发现的网格梯度冲突并未在这次受控筛选中转化为实质提升；不能继续将“分辨率太低”视为已经证实的主要可利用瓶颈，也不能由此否定所有细尺度/端到端方案。

本轮固定1024候选点、单层up1特征、逐点MLP、冻结主干、无增强和有限步数，这些是结论范围。它没有验证候选区域外的高置信错误、空间上下文更强的表示或更早层的文字交互，也没有证明这些方向必然有效。

**处理：封存四组与全部五折负结果，暂停此输出纠错接口的直接扩展；第二步文字训练暂不启动。** 后续需要新的、能说明文字在何类剩余错误上提供额外证据的诊断假设，优先区分表示/定位信息不足与只改变置信度。不能继续在同一OOF结果上挑轮数、阈值或权重，也不自动开80/150轮或Test。本次修复、有限空间和已读文献都不足以把第二创新点判定成立。
'''
(HERE/'REPORT.md').write_text(text,encoding='utf-8',newline='\n')
print(json.dumps(dict(report=str(HERE/'REPORT.md'),next_interface=s['recommended_interface_for_text_screen'])))

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
text+='\n## 细接口的额外作用\n\n'
for v,c in (('fine_free','coarse_free'),('fine_mass','coarse_mass')):
    p=s['comparisons'][v+'-'+c]
    text+=f"- {v}−{c}：ΔIoU {p['delta_iou']*100:+.4f} pp，group CI [{p['group_ci95'][0]*100:+.4f}, {p['group_ci95'][1]*100:+.4f}] pp；细接口门：{s['fine_interface_gates'][v]}。\n"
text+=f'''
预注册规则给出的后续文字对照接口：**{s['recommended_interface_for_text_screen'] or '无候选通过，停止直接推进'}**。该判断不授权改变本轮配置、追加训练、重新调旧折或访问Test。细接口读取的视觉细节与空间耦合同时改变，不把收益全归因于单一分辨率或新增文字；本轮四头没有额外文本输入，原R2仍包含文字。

检查次数{state['inspections_completed']}/2，距全部完成{state.get('completed_unix_inspection_delay_seconds',float('nan'))/60:.3f}分钟。全部样本、完整五折与失败项已报告；只有通过接口门后才按独立协议研究文字增量。当前并不证明稳定Test IoU或第二创新点。
'''
(HERE/'REPORT.md').write_text(text,encoding='utf-8',newline='\n')
print(json.dumps(dict(report=str(HERE/'REPORT.md'),next_interface=s['recommended_interface_for_text_screen'])))

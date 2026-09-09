"""Verify the downloaded completed Test chain, archive it, and update the tracker."""
import json
import shutil
from pathlib import Path
from compare_test import summarize
from protocol import sha256, write_json
from test_control import DOCS, HERE, OUT


def main():
    state = json.loads((OUT / 'state.json').read_text(encoding='utf-8'))
    assert state['phase'] == 'complete'
    for name in ('test_plan.json', 'three_seed_summary.json', 'evaluate_test.py'):
        shutil.copyfile(HERE / name, OUT / name)
    recomputed = summarize(OUT, state['evaluation_source_git_commit'])
    downloaded = json.loads((OUT / 'three_seed_test_summary.json').read_text(encoding='utf-8'))
    assert recomputed == downloaded, 'Downloaded summary differs from independent local recomputation'
    results = HERE / 'results'
    results.mkdir(exist_ok=True)
    for path in OUT.iterdir():
        if path.is_file() and (path.suffix in ('.json', '.log')):
            shutil.copyfile(path, results / path.name)
    lines = ['# R2 三种子 Test 完成结果', '',
             '六个模型各2113张Test图像，固定阈值0.5，按Val macro IoU选Best，所有注册种子均完整报告。', '',
             '| 种子 | C4 IoU | R2 IoU | IoU差值（百分点） | C4 Dice | R2 Dice | Dice差值（百分点） |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for pair in recomputed['paired_runs']:
        c, r, d = pair['control'], pair['candidate'], pair['deltas']
        lines.append(f'| {pair["seed"]} | {c["iou"]*100:.4f}% | {r["iou"]*100:.4f}% | {d["iou"]*100:+.4f} | {c["dice"]*100:.4f}% | {r["dice"]*100:.4f}% | {d["dice"]*100:+.4f} |')
    iou, dice = recomputed['delta_summary']['iou'], recomputed['delta_summary']['dice']
    lines += ['', f'IoU配对差值均值 {iou["mean"]*100:+.4f} 个百分点，样本标准差 {iou["sample_std"]*100:.4f}；Dice差值均值 {dice["mean"]*100:+.4f} 个百分点，样本标准差 {dice["sample_std"]*100:.4f}。', '',
              f'三个种子IoU是否全部正向：{recomputed["all_three_iou_deltas_positive"]}。完整Precision、Recall、Brier与逐图配对区间见JSON。', '',
              f'评估源码：`{state["evaluation_source_git_commit"]}`。完成检查使用{state["inspections_completed"]}/2次，距实际Test完成{state["seconds_after_test_completion"]:.1f}秒，30分钟目标满足：{state["completion_check_within_30min"]}。', '',
              '本次为三个固定训练种子的测试表现。逐图bootstrap不能代替跨种子显著性；该Test集在历史开发中已被访问。R2改变完整学习率配方（重启及较低学习率区间），不能归因于单独一个因素，也不作为第二项结构创新。', '',
              '模型未删除，HF认证旧问题未重试。']
    (results / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    manifest = {p.name: dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in results.iterdir()
                if p.is_file() and p.name != 'artifact_manifest.json'}
    write_json(results / 'artifact_manifest.json', manifest)
    tracker = DOCS / 'docs/EXPERIMENT_TRACKER.md'
    text = tracker.read_text(encoding='utf-8')
    entry = ('## 最新 Test 结果（2026-09-10）\n\n'
             f'R2与C4三个匹配种子的六个Test评估全部完成，各2113张、固定阈值0.5、Val IoU选Best。'
             f'平均IoU配对差值 **{iou["mean"]*100:+.4f}个百分点**（样本SD {iou["sample_std"]*100:.4f}），'
             f'平均Dice差值 **{dice["mean"]*100:+.4f}个百分点**；三个IoU差值全部正向：{recomputed["all_three_iou_deltas_positive"]}。'
             '[全部结果、来源与限制](../repro_archive/20260910/recipe_test/results/README.md)。'
             'R2为训练配方成果，不作为第二项结构创新；没有新增训练。\n\n')
    assert '## 最新 Test 结果（2026-09-10）' not in text, 'Already finalized'
    text = text.replace('## 最新执行记录（2026-09-10）', entry + '## 最新执行记录（2026-09-10）', 1)
    tracker.write_text(text, encoding='utf-8', newline='\n')
    print(json.dumps(dict(complete=True, mean_test_iou_delta=iou['mean'], mean_test_dice_delta=dice['mean'],
                         archive=str(results), all_seeds_positive=recomputed['all_three_iou_deltas_positive'])))


if __name__ == '__main__':
    main()

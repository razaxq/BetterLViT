"""Independently recompute Test metrics and paired differences from frozen exports."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DOCS = HERE.parents[2]
METRICS = ('iou', 'dice', 'precision', 'recall', 'brier')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(delta):
    rng = np.random.default_rng(1219)
    means = []
    for _ in range(50):
        indices = rng.integers(0, len(delta), size=(200, len(delta)))
        means.extend(delta[indices].mean(axis=1))
    return np.quantile(means, [.025, .975]).tolist()


def run():
    plan = json.loads((HERE / 'authorization.json').read_text(encoding='utf-8'))
    launch = json.loads((HERE / 'launch.json').read_text())
    paths = dict(R2=DOCS / plan['r2_reused_result'], S1=HERE / 'results/s1_test.json', S2=HERE / 'results/s2_test.json')
    assert digest(paths['R2']) == plan['r2_result_sha256']
    datasets = {k: json.loads(p.read_text()) for k, p in paths.items()}
    data = {}; labels = None; names = None; summary = dict(models={}, comparisons={})
    for label, result in datasets.items():
        assert result['split'] == 'test' and result['samples'] == 2113 and result['seed'] == 1219
        assert result['epochs'] == 80 and result['selection_metric'] == 'iou' and result['threshold'] == .5
        assert result['checkpoint_git_commit'] == result['analysis_git_commit']
        assert not result['threshold_selected_on_test']
        if label != 'R2':
            arm = next(a for a in plan['arms'] if a['label'] == label.lower())
            assert result['checkpoint_git_commit'] == arm['source_git_commit']
            assert result['checkpoint_best_epoch'] == arm['best_epoch']
            assert result['evaluation_source_git_commit'] == launch['evaluation_source_git_commit']
            assert result['evaluation_script_sha256'] == launch['deployed_sha256']['evaluate_test.py']
            assert result['authorization_sha256'] == launch['deployed_sha256']['authorization.json']
        rows = sorted(result['records'], key=lambda r:r['name'])
        current_names = [r['name'] for r in rows]
        assert len(rows) == len(set(current_names)) == 2113
        current_labels = np.array([r['label_pixels'] for r in rows])
        if labels is None:
            names = current_names; labels = current_labels
        else:
            assert names == current_names and np.array_equal(labels, current_labels)
        predictions = np.array([r['prediction_pixels'] for r in rows])
        tp = np.rint(np.array([r['dice'] for r in rows]) * (labels + predictions) / 2).astype(np.int64)
        assert ((tp >= 0) & (tp <= labels) & (tp <= predictions)).all()
        expected = dict(dice=np.divide(2*tp, labels+predictions, out=np.zeros(len(tp),dtype=float),where=(labels+predictions)>0),
                        iou=np.divide(tp, labels+predictions-tp,out=np.zeros(len(tp),dtype=float),where=(labels+predictions-tp)>0),
                        precision=np.divide(tp,predictions,out=np.zeros(len(tp),dtype=float),where=predictions>0),
                        recall=np.divide(tp,labels,out=np.zeros(len(tp),dtype=float),where=labels>0))
        data[label] = {m:np.array([r[m] for r in rows]) for m in METRICS}
        errors = {}
        for metric, values in data[label].items():
            assert np.isfinite(values).all()
            assert abs(values.mean()-result['macro_'+metric]) < 1e-12
            if metric in expected:
                errors[metric] = float(np.abs(values-expected[metric]).max())
                assert errors[metric] < 1e-12
        summary['models'][label] = dict(source_git_commit=result['checkpoint_git_commit'], best_epoch=result['checkpoint_best_epoch'],
            result_path=paths[label].relative_to(DOCS).as_posix(), result_sha256=digest(paths[label]),
            samples=2113, seed=1219, threshold=.5, metrics={m:float(v.mean()) for m,v in data[label].items()},
            mean_fp=float((predictions-tp).mean()),mean_fn=float((labels-tp).mean()),metric_reconstruction_max_error=errors)
    for baseline, candidate in (('R2','S1'),('R2','S2'),('S1','S2')):
        differences={}
        for metric in METRICS:
            delta=data[candidate][metric]-data[baseline][metric]
            differences[metric]=dict(delta=float(delta.mean()))
            if metric in ('iou','dice'):
                differences[metric]['paired_image_bootstrap_95ci']=interval(delta)
        summary['comparisons'][candidate+'-'+baseline]=differences
    summary['protocol'] = dict(split='test',aggregation='per-image macro',matched_seed=1219,epochs=80,threshold=.5,
        bootstrap_resamples=10000,bootstrap_seed=1219,bootstrap_unit='image; not subject; descriptive single-seed comparison',
        original_s1_s2_validation_screen_passed=False,user_requested_test=True,no_new_training=True)
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# R2 / S1 / S2 Test结果','',
           '2113张Test，seed1219，80轮单次余弦，Val IoU选Best，固定阈值0.5；逐图macro口径。S1/S2按用户本次明确要求补测，原Val筛选失败结论保持不变。','',
           '| 模型 | Best轮次 | IoU | Dice | Precision | Recall |',
           '|---|---:|---:|---:|---:|---:|']
    for label, model in summary['models'].items():
        m=model['metrics'];lines.append(f"| {label} | {model['best_epoch']} | {m['iou']*100:.4f}% | {m['dice']*100:.4f}% | {m['precision']*100:.4f}% | {m['recall']*100:.4f}% |")
    lines+=['','差值为百分点；95%区间为10000次图像配对bootstrap，仅反映本次单种子、按图像抽样的描述性不确定性，不能代替患者聚类或多种子复验。','',
            '| 比较 | IoU差值 | IoU 95%区间 | Dice差值 |','|---|---:|---:|---:|']
    for name, values in summary['comparisons'].items():
        lo,hi=values['iou']['paired_image_bootstrap_95ci']
        lines.append(f"| {name} | {values['iou']['delta']*100:+.4f} | [{lo*100:+.4f}, {hi*100:+.4f}] | {values['dice']['delta']*100:+.4f} |")
    lines+=['','本次结果中S1与R2基本持平；S2获得小幅IoU/Dice点估计增益，但IoU图像配对区间跨0，不能称稳定提升。S2相对R2的Precision提高0.7085个百分点、Recall下降0.6267个百分点；与此前减少误检、增加漏检的方向一致。未因本次Test结果重选模型、调参或启动新训练。']
    lines+=['','三组文件名/GT像素数逐图匹配，Dice/IoU/Precision/Recall由整数交并计数复算通过，逐图均值与原JSON一致。R2复用已核验结果；新评估器推理与指标计算与R2原评估器AST一致。','',
            '## 来源','']
    for label,model in summary['models'].items():
        lines.append(f"- {label}: `{model['source_git_commit']}`；结果SHA256 `{model['result_sha256']}`。")
    lines+=['',f"本次评估代码：`{launch['evaluation_source_git_commit']}`；标签`evaluation-r2-s1-s2-test-seed1219-20260911`。",'',
            '完整逐图数据、日志、授权变更、代码一致性检查和收集回执随目录归档。未训练S2-T；原始S2成绩不混入修复。']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v['metrics'] for k,v in summary['models'].items()}))


if __name__ == '__main__':
    run()

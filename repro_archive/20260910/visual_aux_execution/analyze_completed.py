"""Independently check completed Val records, selection, schedule and Train audits."""
import argparse
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from remote_ops import HERE,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--label',choices=('s1','s2'),required=True);label=p.parse_args().label
    folder=HERE/(label+'_results')
    val=json.loads((folder/'validation.json').read_text())
    base=json.loads((HERE.parents[1]/'20260908/recipe_execution/r2_results/validation.json').read_text())
    history=json.loads((folder/'epoch_history.json').read_text())
    snapshot=json.loads((HERE/(label+'_final_snapshot.json')).read_text())
    manifest=json.loads((HERE/(label+'_manifest.json')).read_text())
    source=json.loads((HERE/'sources.json').read_text())[label]
    assert val['checkpoint_git_commit']==val['analysis_git_commit']==source['source_git_commit']
    assert snapshot['runtime']['training_rc']==snapshot['runtime']['validation_rc']==0
    assert snapshot['runtime_source_clean'] and not val['test_split_accessed']
    assert [r['epoch'] for r in history]==list(range(1,81))
    assert np.allclose([r['lr'] for r in history],manifest['planned_epoch_lrs'],rtol=0,atol=1e-15)
    selected=max(history[5:],key=lambda r:r['val_iou'])
    assert selected['epoch']==val['checkpoint_best_epoch']
    # Training reduces batch metrics in float32; the export uses float64.
    selection_export_difference=selected['val_iou']-val['macro_iou']
    assert abs(selection_export_difference)<1e-7
    for d in (base,val):
        assert len({r['name'] for r in d['records']})==d['samples']==1429
        for key in ('iou','dice','precision','recall','brier'):
            values=np.array([r[key] for r in d['records']]);assert np.isfinite(values).all()
            assert abs(values.mean()-d['macro_'+key])<1e-12
    gate=json.loads((folder/('r2_vs_'+label+'.json')).read_text())
    aa={r['name']:r for r in base['records']};bb={r['name']:r for r in val['records']}
    assert aa.keys()==bb.keys()
    for key in gate['deltas']:
        delta=float(np.mean([bb[n][key]-aa[n][key] for n in sorted(aa)]))
        assert abs(delta-gate['deltas'][key]['mean'])<1e-12
    regional=None;s1=None
    if label=='s2':
        regional=json.loads((folder/'s1_vs_s2.json').read_text())
        s1=json.loads((HERE/'s1_results/validation.json').read_text())
        cc={r['name']:r for r in s1['records']};assert cc.keys()==bb.keys()
        assert regional['control_sha']==s1['checkpoint_git_commit']
        assert regional['candidate_sha']==source['source_git_commit']
        for key in regional['deltas']:
            delta=float(np.mean([bb[n][key]-cc[n][key] for n in sorted(bb)]))
            assert abs(delta-regional['deltas'][key]['mean'])<1e-12
    diagnostics=[];common_names=None
    for epoch in (20,40,60,80):
        d=json.loads((folder/'diagnostics'/f'epoch_{epoch:03d}.json').read_text())
        assert d['source_git_commit']==source['source_git_commit'] and d['epoch']==epoch
        assert d['state_sha256_before']==d['state_sha256_after'] and d['gradient_state_unchanged'] and d['rng_restored']
        names=[r['name'] for b in d['batches'] for r in b['cases']]
        assert len(names)==len(set(names))==32
        if common_names is None:common_names=names
        assert names==common_names
        terms={}
        for term in d['batches'][0]['gradients']:
            if term=='main':continue
            terms[term]={key:np.mean([[r[key] for r in b['gradients'][term]] for b in d['batches']],axis=0).tolist()
                         for key in ('ratio','cosine')}
        diagnostics.append(dict(epoch=epoch,gradients=terms))
    for filename in ('training.log','validation.log'):
        text=(folder/filename).read_text(errors='replace')
        assert not any(x in text for x in ('Traceback (most recent call last)','CUDA out of memory','RuntimeError:'))
    result=dict(verified=True,source_git_commit=source['source_git_commit'],split='validation',test_split_accessed=False,
        samples=1429,epochs=80,best_epoch=val['checkpoint_best_epoch'],actual_lr_matches_manifest=True,
        selected_epoch_matches_full_history=True,selection_export_iou_difference=selection_export_difference,
        macro_summaries_and_deltas_recomputed=True,
        fixed_train_cases=32,diagnostics=diagnostics,passed=gate['passed'],checks=gate['checks'],
        failed_checks=[k for k,v in gate['checks'].items() if not v],
        inspection_number=snapshot['inspection_number'],seconds_after_training_end=snapshot['seconds_after_training_end'],
        within_30_minutes=snapshot['within_30_minutes'])
    if regional is not None:
        result['regional_increment']=dict(passed=regional['passed'],checks=regional['checks'],
            deltas=regional['deltas'],failed_checks=[k for k,v in regional['checks'].items() if not v])
    save(label+'_results/independent_verification.json',result)
    time=lambda x:datetime.fromtimestamp(x,ZoneInfo('Australia/Sydney')).strftime('%Y-%m-%d %H:%M:%S')
    rows=[f'# {label.upper()} Val筛选结果','',f"完成80轮，Val IoU选定epoch {val['checkpoint_best_epoch']}。来源 `{source['source_git_commit']}`；tag `{source['experiment_tag']}`。",'',
        '| 配置 | Val IoU | Dice | Precision | Recall | Brier |','|---|---:|---:|---:|---:|---:|']
    compared=[('R2',base)]+([('S1',s1)] if s1 is not None else [])+[(label.upper(),val)]
    for name,d in compared:
        rows.append('| '+name+' | '+' | '.join(f"{100*d['macro_'+k]:.4f}%" for k in ('iou','dice','precision','recall'))+f" | {d['macro_brier']:.6f} |")
    di=gate['deltas']['iou'];dd=gate['deltas']['dice']
    rows+=['',f"相对R2：IoU {100*di['mean']:+.4f}个百分点，95%图像配对区间[{100*di['ci95'][0]:+.4f}, {100*di['ci95'][1]:+.4f}]；Dice {100*dd['mean']:+.4f}个百分点。筛选通过：{gate['passed']}；失败条件：{', '.join(result['failed_checks']) or '无'}。",'',
        f"总体Dice/precision、最小GT总面积组Dice/recall和Brier条件见[r2_vs_{label}.json](r2_vs_{label}.json)。1429张逐图均值、差值、80轮LR、Best选择与四次固定32张Train诊断独立核验通过。Train梯度为四批eval快照均值，不能直接归因为IoU贡献。",'',
        f"训练结束{time(snapshot['runtime']['training_ended_unix'])}，Val完成{time(snapshot['runtime']['validation_ended_unix'])}，末检{time(snapshot['checked_unix'])}（悉尼）；末检距训练结束{snapshot['seconds_after_training_end']:.3f}秒，检查{snapshot['inspection_number']}/2。",'',
        '当前结果全部为Val，尚无本模型Test成绩。只有通过注册R2门的候选才进入匹配种子复验；区域创新价值还需通过S2对S1的增量门。HF上传完成与否另见上传核验回执。']
    if regional is not None:
        dr=regional['deltas']['iou']
        rows+=['',f"相对S1：IoU {100*dr['mean']:+.4f}个百分点，95%图像配对区间[{100*dr['ci95'][0]:+.4f}, {100*dr['ci95'][1]:+.4f}]；区域增量门通过：{regional['passed']}。失败条件：{', '.join(result['regional_increment']['failed_checks']) or '无'}。",'',
               '这项比较同时加入存在与占比任务，不能区分两者各自作用。未过门则按计划停止当前配置，不延长150轮、不调系数，也不能称第二创新已成功。']
    (folder/'REPORT.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='diagnostics'},ensure_ascii=False))


if __name__=='__main__':main()

"""Recompute completion, schedule, selection, macro metrics and audit provenance."""
import argparse,hashlib,json
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from remote_ops import HERE,save


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',choices=('rs1','rs2','rs3'),required=True)
    label=parser.parse_args().label;folder=HERE/(label+'_results')
    read=lambda name:json.loads((folder/name).read_text(encoding='utf-8'))
    val=read('validation.json');history=read('epoch_history.json');gate=read('r2_vs_'+label+'.json')
    base=json.loads((HERE.parents[1]/'20260908/recipe_execution/r2_results/validation.json').read_text())
    source=json.loads((HERE/'sources.json').read_text())[label]
    manifest=json.loads((HERE/(label+'_manifest.json')).read_text())
    paths=[HERE/(label+'_'+phase+'_snapshot.json') for phase in ('final','first')]
    snapshot=next(json.loads(p.read_text()) for p in paths if p.exists() and json.loads(p.read_text())['runtime']['phase']=='complete')
    assert val['checkpoint_git_commit']==val['analysis_git_commit']==source['source_git_commit']
    assert snapshot['runtime']['training_rc']==snapshot['runtime']['validation_rc']==0 and snapshot['runtime_source_clean']
    assert not val['test_split_accessed'] and val['regional_supervision']['weight']==manifest['regional_weight']
    assert [r['epoch'] for r in history]==list(range(1,81))
    assert np.allclose([r['lr'] for r in history],manifest['planned_epoch_lrs'],rtol=0,atol=1e-15)
    selected=max(history[5:],key=lambda r:r['val_iou'])
    assert selected['epoch']==val['checkpoint_best_epoch']
    selection_delta=val['macro_iou']-selected['val_iou']
    reconciliation_path=folder/'iou_reconciliation.json'
    reconciliation=None
    if reconciliation_path.exists():
        reconciliation=read('iou_reconciliation.json')
        assert reconciliation['verified'] and not reconciliation['test_split_accessed']
        assert reconciliation['source_git_commit']==source['source_git_commit']
        assert reconciliation['checkpoint_best_epoch']==selected['epoch']
        assert reconciliation['samples']==1429
        assert reconciliation['original_export_sha256']==hashlib.sha256((folder/'validation.json').read_bytes()).hexdigest()
        assert reconciliation['training_history_iou']==selected['val_iou']
        assert reconciliation['original_export_iou']==val['macro_iou']
        assert abs(reconciliation['history_residual'])<1e-12 and abs(reconciliation['export_residual'])<1e-12
        assert reconciliation['max_export_record_error']==0
        assert reconciliation['checkpoint_sha256_before']==reconciliation['checkpoint_sha256_after']
        explained=reconciliation['threshold_effect_export_minus_ge']+reconciliation['float32_effect_ge_minus_training']
        assert abs(selection_delta-explained)<1e-12
    else:
        # Preserve the original bound. Larger differences need direct evidence;
        # never widen the tolerance to make a candidate pass.
        assert abs(selection_delta)<1e-7, 'Run reconcile_iou.py on the completed Best; preserve original export'
    aa={r['name']:r for r in base['records']};bb={r['name']:r for r in val['records']}
    assert len(aa)==len(bb)==1429 and aa.keys()==bb.keys()
    for data in (base,val):
        for metric in ('iou','dice','precision','recall','brier'):
            values=np.array([r[metric] for r in data['records']])
            assert np.isfinite(values).all() and abs(values.mean()-data['macro_'+metric])<1e-12
    for metric,item in gate['deltas'].items():
        assert abs(np.mean([bb[n][metric]-aa[n][metric] for n in aa])-item['mean'])<1e-12
    common_names=None
    for epoch in (20,40,60,80):
        d=read('diagnostics/epoch_'+str(epoch).zfill(3)+'.json')
        assert d['epoch']==epoch and d['source_git_commit']==source['source_git_commit']
        assert d['state_sha256_before']==d['state_sha256_after'] and d['gradient_state_unchanged'] and d['rng_restored']
        names=[r['name'] for batch in d['batches'] for r in batch['cases']]
        assert len(names)==len(set(names))==32
        if common_names is None:common_names=names
        assert common_names==names
    for name in ('training.log','validation.log'):
        text=(folder/name).read_text(errors='replace')
        assert not any(s in text for s in ('Traceback (most recent call last)','CUDA out of memory','RuntimeError:'))
    result=dict(verified=True,source_git_commit=source['source_git_commit'],test_split_accessed=False,
        epochs=80,samples=1429,best_epoch=val['checkpoint_best_epoch'],actual_lr_matches_manifest=True,
        macro_summaries_and_deltas_recomputed=True,audits_state_preserving=True,passed=gate['passed'],checks=gate['checks'],
        inspection_number=snapshot['inspection_number'],seconds_after_training_end=snapshot['seconds_after_training_end'],
        within_30_minutes=snapshot['within_30_minutes'],
        export_minus_selection_iou=selection_delta,
        metric_reconciliation_verified=reconciliation is not None)
    if label!='rs1':
        increment=read('rs1_vs_'+label+'.json');result['regional_increment']=increment
    if label=='rs3':
        grouping=read('rs2_vs_rs3.json')
        rs2=json.loads((HERE/'rs2_results/validation.json').read_text())
        parent={r['name']:r for r in rs2['records']}
        assert parent.keys()==bb.keys() and grouping['candidate_sha']==source['source_git_commit']
        assert grouping['control_sha']==rs2['checkpoint_git_commit'] and not grouping['test_split_accessed']
        for metric,item in grouping['deltas'].items():
            assert abs(np.mean([bb[n][metric]-parent[n][metric] for n in parent])-item['mean'])<1e-12
        assert grouping['grouping_iou_ci_positive']==(grouping['deltas']['iou']['ci95'][0]>0)
        result['grouping_attribution']=grouping
    save(label+'_results/independent_verification.json',result)
    delta=gate['deltas']['iou']
    lines=[f'# {label.upper()} Val结果','',f"来源 `{source['source_git_commit']}`，80轮，Best {val['checkpoint_best_epoch']}。",'',
        '| 模型 | IoU | Dice | Precision | Recall | Brier |','|---|---:|---:|---:|---:|---:|']
    table_rows=[('R2',base)]
    if label!='rs1':
        table_rows.append(('RS1',json.loads((HERE/'rs1_results/validation.json').read_text())))
    if label=='rs3':table_rows.append(('RS2',rs2))
    table_rows.append((label.upper(),val))
    for name,data in table_rows:
        lines.append('| '+name+' | '+' | '.join(f"{100*data['macro_'+m]:.4f}%" for m in ('iou','dice','precision','recall'))+f" | {data['macro_brier']:.6f} |")
    lines+=['',f"IoU差值 {100*delta['mean']:+.4f}个百分点，95%图像配对区间 {[(100*v) for v in delta['ci95']]}；R2筛选通过：{gate['passed']}。",'',
        f"末检距训练结束 {snapshot['seconds_after_training_end']:.3f}秒，检查{snapshot['inspection_number']}/2。来源、80轮学习率、Best、逐图均值和四次固定Train诊断已核验。",'',
        '本次结果为Val，无本候选Test成绩。Train诊断为输出logit梯度，不能解释成共享参数梯度冲突或IoU因果贡献。HF备份以独立上传回执为准。']
    lines+=['', '未通过条件：'+('、'.join(k for k,v in gate['checks'].items() if not v) or '无')+'。首轮RS1/RS2/RS3仍按冻结配置全部执行，不根据前组结果调节后组。']
    if reconciliation is not None:
        lines+=['',f"训练历史采用>=0.5及float32批次均值，原逐图导出采用>0.5及float64；同一Best的Val-only复算保留两套原始结果。精确0.5像素{reconciliation['exact_half_pixels']}个（GT阳性{reconciliation['exact_half_positive_pixels']}个），阈值贡献{reconciliation['threshold_effect_export_minus_ge']:.15g}、float32贡献{reconciliation['float32_effect_ge_minus_training']:.15g}，完全解释导出减历史{selection_delta:.15g}。逐图导出及训练IoU复算残差均小于1e-12，检查点SHA256前后相同；见iou_reconciliation.json。"]
    if label!='rs1':
        inc=increment['deltas']['iou']
        lines+=['',f"相对RS1的IoU差值{100*inc['mean']:+.4f}个百分点，95%图像配对区间{[100*v for v in inc['ci95']]}；区域增量门通过：{increment['passed']}，完整差值见rs1_vs_{label}.json。",
            '',f"按R2与当前候选共同GT面积最低四分位（≤{gate['small_area_cutoff']:.0f}像素，{gate['small_count']}张）比较：IoU {100*gate['deltas']['iou']['small_mean']:+.4f}、Dice {100*gate['deltas']['dice']['small_mean']:+.4f}、Recall {100*gate['deltas']['recall']['small_mean']:+.4f}个百分点。这里只是按标注面积分层，不代表临床病灶分型。"]
    if label=='rs3':
        g=grouping['deltas']['iou']
        lines+=['',f"RS3−RS2分组归因比较：IoU {100*g['mean']:+.4f}个百分点，95%配对区间{[100*v for v in g['ci95']]}；区间下界>0：{grouping['grouping_iou_ci_positive']}。此比较只用于判断分组的独立收益，不替代R2性能门或RS1区域增量门。"]
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()

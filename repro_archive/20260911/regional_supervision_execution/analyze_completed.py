"""Recompute completion, schedule, selection, macro metrics and audit provenance."""
import argparse,json
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
    assert selected['epoch']==val['checkpoint_best_epoch'] and abs(selected['val_iou']-val['macro_iou'])<1e-7
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
        within_30_minutes=snapshot['within_30_minutes'])
    if label!='rs1':
        increment=read('rs1_vs_'+label+'.json');result['regional_increment']=increment
    save(label+'_results/independent_verification.json',result)
    delta=gate['deltas']['iou']
    lines=[f'# {label.upper()} Val结果','',f"来源 `{source['source_git_commit']}`，80轮，Best {val['checkpoint_best_epoch']}。",'',
        '| 模型 | IoU | Dice | Precision | Recall | Brier |','|---|---:|---:|---:|---:|---:|']
    for name,data in (('R2',base),(label.upper(),val)):
        lines.append('| '+name+' | '+' | '.join(f"{100*data['macro_'+m]:.4f}%" for m in ('iou','dice','precision','recall'))+f" | {data['macro_brier']:.6f} |")
    lines+=['',f"IoU差值 {100*delta['mean']:+.4f}个百分点，95%图像配对区间 {[(100*v) for v in delta['ci95']]}；R2筛选通过：{gate['passed']}。",'',
        f"末检距训练结束 {snapshot['seconds_after_training_end']:.3f}秒，检查{snapshot['inspection_number']}/2。来源、80轮学习率、Best、逐图均值和四次固定Train诊断已核验。",'',
        '本次结果为Val，无本候选Test成绩。Train诊断为输出logit梯度，不能解释成共享参数梯度冲突或IoU因果贡献。HF备份以独立上传回执为准。']
    if label!='rs1':lines+=['',f"相对RS1的区域增量门通过：{increment['passed']}，完整差值见rs1_vs_{label}.json。"]
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()

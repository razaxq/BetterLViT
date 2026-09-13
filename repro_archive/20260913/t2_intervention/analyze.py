"""Independently recompute integer metrics and descriptive paired intervals."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'text_decoder_controls'))
from analyze import interval as clustered_interval

METRICS=('iou','dice','precision','recall','brier')

def main():
    d=json.loads((HERE/'result.json').read_text())
    assert d['status']=='complete' and not d['test_split_accessed'] and not d['training_performed']
    assert d['historical_counts_exact'] and d['model_state_unchanged'] and d['repeat_output_exact']
    data={k:{r['name']:r for r in rows} for k,rows in d['per_mode'].items()}
    names=sorted(data['real'])
    donors={r['name']:r for r in d['donor_map']}
    assert len(names)==1429 and set(names)==set(donors)
    small=np.array([data['real'][n]['label_pixels']<=2075 for n in names])
    changed=np.array([donors[n]['changed'] for n in names])
    for mode,rows in data.items():
        assert set(rows)==set(names)
        for n,r in rows.items():
            t,p,g=r['tp'],r['prediction_pixels'],r['label_pixels']
            assert g==data['real'][n]['label_pixels'] and 0<=t<=min(p,g)<=50176
            expected=dict(iou=t/(p+g-t) if p+g-t else 0.,dice=2*t/(p+g) if p+g else 0.,
                precision=t/p if p else 0.,recall=t/g if g else 0.)
            assert all(abs(r[k]-v)<1e-12 for k,v in expected.items())
        for m in METRICS:
            assert abs(np.mean([r[m] for r in rows.values()])-d['summaries'][mode][m])<1e-12
    contrasts={}
    for cand,base in [('off','real'),('mismatch','real'),('mismatch','off')]:
        key=cand+'_minus_'+base
        contrasts[key]={}
        for label,mask in [('all',np.ones(len(names),dtype=bool)),('small',small),('eligible_mismatch',changed),('small_eligible_mismatch',small&changed)]:
            nn=[n for n,yes in zip(names,mask) if yes]
            if not nn:continue
            v={m:np.array([data[cand][n][m]-data[base][n][m] for n in nn]) for m in METRICS}
            contrasts[key][label]=dict(samples=len(nn),metric_deltas={m:dict(mean=float(x.mean()),**clustered_interval(x,nn)) for m,x in v.items()},
                pixel_count_deltas={m:sum(data[cand][n][m]-data[base][n][m] for n in nn) for m in ('tp','fp','fn')})
    off=contrasts['off_minus_real']['small']
    supports=off['metric_deltas']['iou']['mean']>0 and off['pixel_count_deltas']['fp']<0
    report=dict(verified=True,result_sha256=hashlib.sha256((HERE/'result.json').read_bytes()).hexdigest(),
        samples=len(names),mismatch_eligible=int(changed.sum()),small_samples=int(small.sum()),
        contrasts=contrasts,off_reduces_small_lesion_harm_directionally=bool(supports),
        interpretation='Post-hoc frozen-checkpoint interventions, not R2 retraining or independent seed evidence',
        next_action='Develop minimal phrase and null-match pair' if supports else 'Investigate representation and optimization before phrase/null-match training')
    (HERE/'analysis.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['# T2干预完成：第二创新点机制筛选','',
        '固定T2 Best69；1429张Val、严格>0.5；原有文字路径始终真实。不训练、不访问Test。',
        '真实文字逐图TP/预测面积/GT面积与历史结果精确一致；参数与buffer前后哈希、首批重复输出均一致。','',
        '| 新增分支 | macro IoU | Dice | Precision | Recall |','|---|---:|---:|---:|---:|']
    for mode,label in [('real','真实文字'),('off','关闭'),('mismatch','同长度错配')]:
        s=d['summaries'][mode]
        lines.append('| '+label+' | '+' | '.join(f'{s[m]*100:.4f}%' for m in METRICS[:4])+' |')
    lines+=['',f"错配可干预{int(changed.sum())}/{len(names)}张；无不同同长度报告的样本保持原文，另报可干预子集。",'',
        '| 相对真实文字 | 全体ΔIoU pp | 最小病灶ΔIoU pp | 最小病灶ΔFP像素 |','|---|---:|---:|---:|']
    for mode in ('off','mismatch'):
        c=contrasts[mode+'_minus_real'];lines.append(f"| {mode} | {c['all']['metric_deltas']['iou']['mean']*100:+.4f} | {c['small']['metric_deltas']['iou']['mean']*100:+.4f} | {c['small']['pixel_count_deltas']['fp']:+d} |")
    lines+=['',f"关闭分支缓解小病灶损害的方向性条件：{supports}。",'',
        '关闭分支不等于重新训练的R2；错配变化不能独立证明语义理解。区间按患者/文件分组描述，不替代跨训练种子验证。',
        '后续：'+report['next_action']+'。所有对照、区间和逐图整数复算见analysis.json。']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(verified=True,summaries=d['summaries'],next_action=report['next_action'],small_off=off)))

if __name__=='__main__':main()

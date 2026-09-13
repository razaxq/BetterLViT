"""Independently recompute counts and freeze Val choices before Test."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):(HERE/n).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8',newline='\n')


def recompute(result):
    assert result['completed'] and result['no_training']
    assert result['samples']==len(result['names'])==len(set(result['names']))
    assert result['plan_sha256']==sha(HERE/'plan.json')
    assert all(p['exact_per_image_counts'] for p in result['parity'].values())
    labels=np.array(result['label_pixels'],dtype=np.int64)[:,None]
    outputs={}
    for name,data in result['modes'].items():
        tp=np.array(data['tp'],dtype=np.int64);pred=np.array(data['pred'],dtype=np.int64)
        assert tp.shape==pred.shape==(result['samples'],len(data['thresholds']))
        assert (tp>=0).all() and (tp<=pred).all() and (tp<=labels).all() and (pred<=224*224).all()
        values={}
        for metric,num,denom in [('iou',tp,pred+labels-tp),('dice',2*tp,pred+labels),
                                 ('precision',tp,pred),('recall',tp,labels)]:
            val=np.divide(num,denom,out=np.zeros_like(tp,dtype=float),where=denom>0)
            assert np.allclose(val.mean(0),data['macro'][metric],rtol=0,atol=1e-12)
            values[metric]=val
        outputs[name]=values
    return outputs


def interval(delta,names):
    ids=[m.group(1) if (m:=re.search(r'(sub-S\d+)',n)) else 'file:'+n for n in names]
    unique=sorted(set(ids));lookup={g:i for i,g in enumerate(unique)}
    index=np.array([lookup[g] for g in ids]);size=np.bincount(index);sums=np.bincount(index,weights=delta)
    rng=np.random.default_rng(1219);samples=[]
    for _ in range(50):
        draw=rng.integers(len(unique),size=(200,len(unique)))
        samples.append(sums[draw].sum(1)/size[draw].sum(1))
    return dict(ci95=np.quantile(np.concatenate(samples),[.025,.975]).tolist(),groups=len(unique),resamples=10000,
                interpretation='Descriptive clustered image interval, not independent training seeds; Val selection optimism remains')


def main():
    p=argparse.ArgumentParser();p.add_argument('split',choices=('validation','test'));a=p.parse_args()
    path=HERE/a.split/'result.json';result=read(path)
    assert read(HERE/a.split/'download_verified.json')['verified']
    assert result['split']==a.split and result['test_split_accessed']==(a.split=='test')
    assert result['samples']==(1429 if a.split=='validation' else 2113)
    vals=recompute(result);method='r2s1219';data=result['modes'][method]
    thresholds=data['thresholds'];idx0=thresholds.index(.5)
    values=vals[method];labels=np.array(result['label_pixels'])
    small=labels<=np.quantile(labels,.25)
    if a.split=='validation':
        assert not (HERE/'selection.json').exists(), 'Never replace a frozen selection'
        idx=sorted(range(len(thresholds)),key=lambda i:(-data['macro']['iou'][i],abs(thresholds[i]-.5),thresholds[i]))[0]
    else:
        selection=read(HERE/'selection.json');assert selection['frozen'] and selection['methods']
        assert result['selection_sha256']==sha(HERE/'selection.json')
        idx=thresholds.index(selection['methods'][method]['threshold'])
    deltas={m:values[m][:,idx]-values[m][:,idx0] for m in values}
    delta={m:float(v.mean()) for m,v in deltas.items()}
    small_delta={m:float(v[small].mean()) for m,v in deltas.items()}
    bootstrap=interval(deltas['iou'],result['names'])
    gain=delta['iou'];passed=gain>=.001 and small_delta['iou']>=0
    summary=dict(verified=True,split=a.split,source_git_commit=result['evaluation_source_git_commit'],
        checkpoint_source_git_commit=result['checkpoint_provenance'][0]['source_git_commit'],
        result_sha256=sha(path),threshold=thresholds[idx],baseline_threshold=.5,
        baseline={m:float(v[:,idx0].mean()) for m,v in values.items()},
        candidate={m:float(v[:,idx].mean()) for m,v in values.items()},
        delta=delta,small_delta=small_delta,small_count=int(small.sum()),small_area_cutoff=float(np.quantile(labels,.25)),
        bootstrap=bootstrap,test_split_accessed=result['test_split_accessed'],no_extra_parameters=True,
        no_extra_deployed_forward_passes=True,novelty_claim=False)
    if a.split=='validation':
        item=dict(threshold=thresholds[idx],passed_validation_gate=bool(passed),macro_iou_delta=gain,small_iou_delta=small_delta['iou'])
        selection=dict(frozen=True,plan_sha256=sha(HERE/'plan.json'),validation_result_sha256=sha(path),
            evaluation_source_git_commit=result['evaluation_source_git_commit'],methods={method:item} if passed else {},
            rejected={} if passed else {method:item},test_split_accessed=False)
        save('selection.json',selection)
        summary['passed_validation_gate']=bool(passed)
    save(a.split+'/summary.json',summary)
    lines=[f'# R2单模型阈值：{a.split}', '',
        f"来源 `{summary['checkpoint_source_git_commit']}`；seed1219，Best67；{result['samples']}张逐图macro。",
        '模型参数和部署前向次数不变；无训练、集成、TTA、连通区域过滤。收益属于阈值工程优化，不是第二创新点。','',
        '| 设置 | 阈值 | IoU | Dice | Precision | Recall |','|---|---:|---:|---:|---:|---:|']
    for name,key,threshold in [('原R2','baseline',.5),('Val选定阈值','candidate',summary['threshold'])]:
        lines.append(f'| {name} | {threshold:.2f} | '+' | '.join(f'{summary[key][m]*100:.4f}%' for m in ('iou','dice','precision','recall'))+' |')
    ci=bootstrap['ci95']
    lines+=['',f"IoU变化{gain*100:+.4f} pp，分组描述性95%区间[{ci[0]*100:+.4f}, {ci[1]*100:+.4f}] pp。",
        f"最小面积{summary['small_count']}张：IoU {small_delta['iou']*100:+.4f} pp，Dice {small_delta['dice']*100:+.4f} pp，Recall {small_delta['recall']*100:+.4f} pp。",
        '', '所有原0.5逐图TP/预测面积/真值面积与历史导出精确一致；整数计数及macro已在本地独立复算。']
    if a.split=='validation':
        lines+=['',f'预登记Test推进门通过={passed}。阈值在Val选择，Val增益存在选择乐观偏差，不能当作Test或稳定性证据。']
    else:
        lines+=['','阈值在此次Test前已按Val冻结；未在Test搜索或改阈值。该Test历史已被访问；单个训练种子结果不等于跨种子稳定收益。']
    (HERE/a.split/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()

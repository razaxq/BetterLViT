"""Analyze relation and local-gradient probes without selecting a model."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'text_decoder_controls'))
from analyze import interval

def main():
    original=json.loads((HERE/'result.json').read_text())
    probe=json.loads((HERE/'binding_result.json').read_text())
    assert probe['status']=='complete' and probe['model_state_unchanged'] and probe['optimizer_steps']==0
    base={r['name']:r for r in original['per_mode']['real']}
    wrong={r['name']:r for r in probe['records']}
    random={r['name']:r for r in original['per_mode']['mismatch']}
    mapping={r['name']:r for r in probe['mapping']}
    assert set(base)==set(wrong)==set(mapping)
    eligible=sorted(n for n,m in mapping.items() if m['changed'])
    for n,r in wrong.items():
        t,p,g=r['tp'],r['prediction_pixels'],r['label_pixels']
        assert g==base[n]['label_pixels']
        assert abs(r['iou']-t/(p+g-t))<1e-12
        if not mapping[n]['changed']:assert r['tp']==base[n]['tp'] and p==base[n]['prediction_pixels']
    contrasts={}
    for label,names in [('eligible',eligible),('small_eligible',[n for n in eligible if base[n]['label_pixels']<=2075])]:
        if not names:continue
        contrasts[label]={}
        for mode,rows in [('binding',wrong),('generic',random)]:
            metrics={m:np.array([rows[n][m]-base[n][m] for n in names]) for m in ('iou','dice','precision','recall','brier')}
            contrasts[label][mode]=dict(samples=len(names),deltas={m:dict(mean=float(x.mean()),**interval(x,names)) for m,x in metrics.items()},
                pixel_deltas={m:sum(rows[n][m]-base[n][m] for n in names) for m in ('tp','fp','fn')})
    g=probe['train_gradient_records']
    values=np.array([x['cosine_residual_vs_bypass'] for x in g]);ratio=np.array([x['residual_gradient_norm_over_bypass'] for x in g])
    assert len(g)==64 and np.isfinite(values).all() and np.isfinite(ratio).all()
    gradients=dict(samples=len(g),negative_count=int((values<0).sum()),mean_cosine=float(values.mean()),
        median_cosine=float(np.median(values)),mean_norm_ratio=float(ratio.mean()),max_norm_ratio=float(ratio.max()))
    result=dict(verified=True,eligible_binding=len(eligible),contrasts=contrasts,gradients=gradients,
        artifact_sha256={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('result.json','binding_result.json')},
        test_split_accessed=False,training_performed=False,
        interpretation='Text binding sensitivity and local eval-mode Jacobian effects; neither causal optimization proof nor new architecture evidence')
    (HERE/'binding_analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    lines=['# 短语绑定及局部梯度探针','',f'原Val中有{len(eligible)}张可找到同有效WordPiece多重集合、不同位置绑定的真实报告供体。其他图像保持原文，逐图计数精确不变。','',
        '| 子集 | 供体类型 | 样本数 | ΔIoU pp | ΔDice pp | ΔPrecision pp | ΔRecall pp |','|---|---|---:|---:|---:|---:|---:|']
    for label,c in contrasts.items():
        for mode,r in c.items():lines.append(f"| {label} | {mode} | {r['samples']} | "+' | '.join(f"{r['deltas'][m]['mean']*100:+.4f}" for m in ('iou','dice','precision','recall'))+' |')
    lines+=['',f"固定Train64张的残差梯度与旁路梯度：负内积{gradients['negative_count']}/64，平均cosine {gradients['mean_cosine']:.6f}，平均范数比 {gradients['mean_norm_ratio']:.6f}。两种反向路径的前向输出完全相同，权重未更新。",'',
        '供体选择不读取图像、mask或模型预测。错误报告仍使用原图真值评估，只作输入干预；不生成伪造目标。',
        'WordPiece词袋相同不保证除位置绑定外所有语言属性完全相同。严格语法与关系记录可追溯，但不能认领纯因果干预。',
        '梯度来自已训练模型的eval模式和局部解码特征，负内积不直接证明原训练冲突或主干退化。',
        '本报告不启动候选训练，不根据诊断结果修改阈值或选择新的正式模型。']
    (HERE/'BINDING_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()

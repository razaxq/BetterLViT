"""Build the three-run summary only from completed, independently verified evidence."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sources=read(HERE/'sources.json')
    baseline_path=HERE.parents[1]/'20260908/recipe_execution/r2_results/validation.json'
    baseline=read(baseline_path)
    rows=[]
    for label in ('rs1','rs2','rs3'):
        folder=HERE/(label+'_results')
        val=read(folder/'validation.json');proof=read(folder/'independent_verification.json')
        upload=read(folder/'hf_upload_verified.json');download=read(folder/'download_xet_verified.json')
        gate=read(folder/('r2_vs_'+label+'.json'));runtime=read(folder/'runtime.json')
        source=sources[label]['source_git_commit']
        assert all(x['verified'] and x['source_git_commit']==source for x in (proof,upload,download))
        assert val['checkpoint_git_commit']==source and runtime['phase']=='complete'
        assert runtime['training_rc']==runtime['validation_rc']==0 and not val['test_split_accessed']
        assert proof['epochs']==80 and proof['samples']==1429 and proof['within_30_minutes']
        r=dict(label=label,source_git_commit=source,best_epoch=val['checkpoint_best_epoch'],
            metrics={m:val['macro_'+m] for m in ('iou','dice','precision','recall','brier')},
            versus_r2=gate,validation_independently_verified=True,hf_upload_verified=True,
            hf_download_verified=True,hf_files=upload['verified_files'],hf_bytes=upload['logical_bytes'],
            inspection_count=proof['inspection_number'],seconds_after_training_end=proof['seconds_after_training_end'],
            training_ended_sydney=datetime.fromtimestamp(runtime['training_ended_unix'],ZoneInfo('Australia/Sydney')).isoformat(),
            input_sha256={p.name:digest(p) for p in (folder/'validation.json',folder/'independent_verification.json',folder/'hf_upload_verified.json',folder/'download_xet_verified.json')})
        if label!='rs1':r['versus_rs1']=read(folder/('rs1_vs_'+label+'.json'))
        if label=='rs3':r['versus_rs2']=read(folder/'rs2_vs_rs3.json')
        rows.append(r)
    result=dict(classification='completed_single_seed_validation_only_first_round',test_split_accessed=False,
        baseline_source_git_commit=baseline['checkpoint_git_commit'],baseline_sha256=digest(baseline_path),
        runs=rows,all_registered_overall_gates_failed=all(not r['versus_r2']['passed'] for r in rows),
        next_step='Preserve RS1 positive Val IoU evidence; no RS4 or epoch extension. Continue the authorized text-path diagnostic A, then evidence-gated B.')
    assert result['all_registered_overall_gates_failed']
    (HERE/'first_round_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# RS1 / RS2 / RS3 首轮结果','',
        '三组均80轮、seed1219、batch16、R2单次余弦及相同主损失；全部训练/Val退出码0，来源、80轮学习率、Best与逐图指标、Train诊断状态恢复及HF备份已核验。本阶段没有Test结果。','',
        '| 模型 | Val IoU | Val Dice | Precision | Recall | IoU相对R2 | Best | 综合门 |',
        '|---|---:|---:|---:|---:|---:|---:|---|',
        '| R2 | '+' | '.join(f'{100*baseline["macro_"+m]:.4f}%' for m in ('iou','dice','precision','recall'))+' | — | 67 | 参照 |']
    for row in rows:
        gate=row['versus_r2'];m=row['metrics']
        lines.append('| '+row['label'].upper()+' | '+' | '.join(f'{100*m[k]:.4f}%' for k in ('iou','dice','precision','recall'))+
            f" | {100*gate['deltas']['iou']['mean']:+.4f} pp | {row['best_epoch']} | 未通过 |")
    lines+=['','## 配对证据与解释','',
        '下列区间为10000次图像配对bootstrap的95%区间。它们描述该单种子检查点的图像抽样差异，不能替代多种子训练稳定性或最终Test。','']
    for row in rows:
        gate=row['versus_r2'];d=gate['deltas']['iou'];ci=d['ci95']
        failed='、'.join(k for k,v in gate['checks'].items() if not v)
        lines.append(f"- {row['label'].upper()}−R2：IoU {100*d['mean']:+.4f} pp，区间[{100*ci[0]:+.4f}, {100*ci[1]:+.4f}]；失败项：{failed}。")
    g=rows[-1]['versus_rs2']['deltas'];d=g['iou'];p=g['precision'];r=g['recall']
    lines+=['',f"RS3−RS2：IoU {100*d['mean']:+.4f} pp，区间[{100*d['ci95'][0]:+.4f}, {100*d['ci95'][1]:+.4f}]。Precision {100*p['mean']:+.4f} pp、Recall {100*r['mean']:+.4f} pp；表现为召回提高同时精度降低，没有证明分组设计带来独立IoU收益。",'',
        'RS1仍有正向的单种子Val IoU证据（+0.4111 pp、图像配对区间下界>0），但Precision下降导致既定综合门失败。应保留这条证据，不能把“三组未通过综合门”写成“三组IoU都没有提高”；也不能将其称为稳定Test增益。','',
        '首轮按冻结顺序完成，不按前组结果修改后组。依用户文字主导的研究方向接续A通路诊断，随后按证据注册B小分支对照；不启动RS4、不延长150轮、不用Test选模块。','',
        '## 来源、检查和备份','']
    for row in rows:
        lines.append(f"- {row['label'].upper()}：`{row['source_git_commit']}`，检查{row['inspection_count']}/2，末检距结束{row['seconds_after_training_end']/60:.2f}分钟；HF `{row['source_git_commit'][:8]}/` {row['hf_files']}文件、{row['hf_bytes']:,}字节，完整清单和下载产物哈希均验证。")
    lines+=['','每组完整报告、所有差值和失败项分别见`rs1_results/REPORT.md`、`rs2_results/REPORT.md`、`rs3_results/REPORT.md`。']
    (HERE/'FIRST_ROUND_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='runs'},ensure_ascii=False))


if __name__=='__main__':main()

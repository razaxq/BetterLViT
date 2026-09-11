"""Create a fixed optimizer diagnosis manifest and adapt the bounded controller."""
import json
from pathlib import Path
from analysis import write_json,digest
HERE=Path(__file__).resolve().parent;B=HERE.parent/'text_head_screening'
m=json.loads((B/'manifest.json').read_text())
manifest=dict(phase='D_train_only_optimizer_collapse_diagnosis',seed=1219,steps=512,batch_size=16,
    completed_b_directory='/root/text_head_b_49905dbb',completed_b_source_git_commit='49905dbbdc644454a37a4a49098db0d5df5fe75a',
    baseline_source_git_commit=m['baseline_source_git_commit'],heads_sha256=digest(HERE/'heads.py'),
    initial_state_sha256=json.loads((B/'results/provenance.json').read_text())['initial_state_sha256'],
    b_artifact_sha256={p:digest(B/p) for p in ('manifest.json','split.json','results/training_order.json','results/cache_manifest.json','results/train_diagnostics.json')},
    treatments=['adam_l2','adam_zero_decay','adamw'],heads_per_treatment=6,lr_schedule_original_steps=1024,
    cache_budget_bytes=0,minimum_free_after_cache_bytes=128_000_000,maximum_inspections=2,
    first_check_delay_seconds=480,completion_check_margin_seconds=180,
    estimated_compute_seconds_from_B=512*.15298429504036903*3+60,
    official_validation_accessed=False,test_split_allowed=False,internal_holdout_evaluated=False,
    no_automatic_architecture_pass=True,train_diagnostic_steps=[0,64,256,512])
write_json(HERE/'manifest.json',manifest)
source=(B/'control.py').read_text()
source=source.replace('from screen_analysis import summarize\n','')
start=source.index('FILES=');end=source.index("\nM=json.loads",start)
source=source[:start]+"FILES=('run_screen.py','heads.py','check_heads.py','analysis.py','control.py','manifest.json','PROTOCOL.md')"+source[end:]
source=source.replace("root='/root/text_head_b_'+sha[:8]","root='/root/text_optimizer_d_'+sha[:8]")
source=source.replace("assert json.loads(a.read_text())['verified']","assert json.loads(a.read_text())['verified']\n    b=HERE.parent/'text_head_screening/results/independent_verification.json'\n    assert json.loads(b.read_text())['verified']")
start=source.index('def verify():');end=source.index("\n\nif __name__=='__main__':",start)
source=source[:start]+'''def verify():
    import numpy as np
    target=HERE/'results';runtime=json.loads((target/'runtime.json').read_text())
    assert runtime['phase']=='complete'
    for name,meta in runtime['artifacts'].items():
        assert (target/name).stat().st_size==meta['bytes'] and digest(target/name)==meta['sha256'],name
    summary=json.loads((target/'summary.json').read_text())
    assert summary['source_git_commit']==runtime['source_git_commit']
    assert summary['original_adam_0_256_512_exact_reproduction']
    assert not summary['internal_holdout_evaluated'] and not summary['test_split_accessed']
    history=json.loads((target/'history.json').read_text())
    assert [r['step'] for r in history]==list(range(1,513))
    cases=json.loads((target/'train_diagnostics.json').read_text())[-1]['cases']
    assert len(cases)==18 and len(summary['cases'])==18
    fit=set(json.loads((HERE.parent/'text_head_screening/results/training_order.json').read_text())['fit_indices'])
    for key,value in cases.items():
        rs=value['records'];assert len(rs)==32 and all(r['index'] in fit for r in rs)
        for row in rs:
            for metric in (row['baseline'],row['output']):
                tp,fp,fn=metric['tp'],metric['fp'],metric['fn']
                assert metric['iou']==(tp/(tp+fp+fn) if tp+fp+fn else 0.)
                assert metric['dice']==(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
        assert summary['cases'][key]['delta_iou']==float(np.mean([r['output']['iou']-r['baseline']['iou'] for r in rs]))
        assert summary['cases'][key]['mean_abs_delta']==float(np.mean([r['mean_abs_delta'] for r in rs]))
    proof=dict(verified=True,source_git_commit=runtime['source_git_commit'],cases=18,steps_per_case=512,
        train_diagnostic_samples=32,artifact_hashes_verified=True,summary_recomputed=True,
        original_adam_reproduction=True,internal_holdout_evaluated=False,test_split_accessed=False)
    write_json(target/'independent_verification.json',proof);print(json.dumps(proof))
'''+source[end:]
(HERE/'control.py').write_text(source,encoding='utf-8',newline='\n')
print(json.dumps(manifest))

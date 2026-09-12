"""IoU-first paired replication analysis; only reads completed immutable Val exports."""
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np
from remote_ops import HERE,DOCS,read,save
METRICS=('iou','dice','precision','recall','brier')

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def groups(names):
    return [(m.group(1) if (m:=re.search(r'(sub-S\d+)',n)) else 'file:'+n) for n in names]

def interval(values,names):
    """Sample clusters uniformly; retain the macro-per-image estimand in each resample."""
    ids=groups(names);unique=sorted(set(ids));lookup={g:i for i,g in enumerate(unique)}
    index=np.array([lookup[g] for g in ids]);sizes=np.bincount(index)
    sums=np.bincount(index,weights=np.asarray(values,dtype=float))
    rng=np.random.default_rng(1219);boots=[]
    for _ in range(50):
        draw=rng.integers(len(unique),size=(200,len(unique)))
        boots.append(sums[draw].sum(1)/sizes[draw].sum(1))
    return dict(ci95=np.quantile(np.concatenate(boots),[.025,.975]).tolist(),groups=len(unique),
        explicit_patient_groups=sum(not g.startswith('file:') for g in unique),resamples=10000,bootstrap_seed=1219,
        interpretation='Descriptive grouped image interval, not across-training-seed significance')

def validate_export(data):
    assert data['split']=='validation' and not data['test_split_accessed']
    assert len(data['checkpoint_git_commit'])==40 and data['analysis_git_commit']==data['checkpoint_git_commit']
    assert data['samples']==len(data['records'])==1429 and data['epochs']==80 and data['threshold']==.5
    assert data['selection_metric']=='iou' and not data['text_use_lora'] and data['boundary_loss_weight']==0
    assert data['training_recipe']['lr_schedule']=='single_cosine' and data['training_recipe']['augmentation_policy']=='legacy'
    records={r['name']:r for r in data['records']};assert len(records)==1429
    for metric in METRICS:
        a=np.array([r[metric] for r in data['records']]);assert np.isfinite(a).all()
        assert abs(a.mean()-data['macro_'+metric])<1e-12
    for r in records.values():
        label=r['label_pixels'];pred=r['prediction_pixels'];tp=round(r['recall']*label)
        assert abs(tp-r['recall']*label)<1e-7 and 0<=tp<=min(label,pred)<=224*224
        expected=dict(iou=tp/(pred+label-tp) if pred+label-tp else 0,
            dice=2*tp/(pred+label) if pred+label else 0,precision=tp/pred if pred else 0,recall=tp/label if label else 0)
        for k,v in expected.items():assert abs(r[k]-v)<1e-12,(r['name'],k)
    return records

def compare(base,candidate):
    a=validate_export(base);b=validate_export(candidate)
    assert base['experiment']=='r2_single_cosine' and candidate['experiment']=='rs1_global_iou'
    assert base['seed']==candidate['seed'] and a.keys()==b.keys()
    names=sorted(a);areas=np.array([a[n]['label_pixels'] for n in names]);assert np.array_equal(areas,[b[n]['label_pixels'] for n in names])
    small=areas<=np.quantile(areas,.25);deltas={}
    for m in METRICS:
        delta=np.array([b[n][m]-a[n][m] for n in names])
        deltas[m]=dict(mean=float(delta.mean()),small_mean=float(delta[small].mean()),**interval(delta,names))
    counts={}
    for label,rows in (('r2',a),('rs1',b)):
        tp=sum(round(r['recall']*r['label_pixels']) for r in rows.values())
        counts[label]=dict(tp=tp,fp=sum(r['prediction_pixels'] for r in rows.values())-tp,
            fn=sum(r['label_pixels'] for r in rows.values())-tp)
    counts['delta']={k:counts['rs1'][k]-counts['r2'][k] for k in ('tp','fp','fn')}
    return dict(seed=base['seed'],control_sha=base['checkpoint_git_commit'],candidate_sha=candidate['checkpoint_git_commit'],
        baseline={m:base['macro_'+m] for m in METRICS},candidate={m:candidate['macro_'+m] for m in METRICS},
        deltas=deltas,pixel_counts_reconstructed_and_metrics_verified=counts,
        small_area_cutoff=float(np.quantile(areas,.25)),small_count=int(small.sum()),split='validation',test_split_accessed=False,
        individual_seed_pass_is_not_overall_replication=True)

def verify_completed(label):
    source=read(HERE/'sources.json')[label];folder=HERE/(label+'_results');val=read(folder/'validation.json')
    baseline=DOCS/source['baseline_validation_relative'];assert digest(baseline)==source['baseline_validation_sha256']
    base=read(baseline);assert base['checkpoint_git_commit']==source['baseline']['source_git_commit']
    assert val['checkpoint_git_commit']==source['source_git_commit'] and val['seed']==source['seed']
    assert val['regional_supervision']['mode']=='global' and val['regional_supervision']['weight']==.128312
    comparison=compare(base,val);manifest=read(HERE/(label+'_manifest.json'));history=read(folder/'epoch_history.json')
    assert [r['epoch'] for r in history]==list(range(1,81))
    assert np.allclose([r['lr'] for r in history],manifest['planned_epoch_lrs'],rtol=0,atol=1e-15)
    selected=max(history[5:],key=lambda r:r['val_iou']);assert selected['epoch']==val['checkpoint_best_epoch']
    selection_delta=val['macro_iou']-selected['val_iou'];reconciliation=None
    if abs(selection_delta)>=1e-7 or (folder/'iou_reconciliation.json').exists():
        assert (folder/'iou_reconciliation.json').exists(),'Use reconcile_iou.py --label '+label+' on the completed Best; preserve original export'
        reconciliation=read(folder/'iou_reconciliation.json')
        assert reconciliation['verified'] and not reconciliation['test_split_accessed']
        assert reconciliation['source_git_commit']==source['source_git_commit']
        assert reconciliation['original_export_sha256']==digest(folder/'validation.json')
        assert reconciliation['checkpoint_best_epoch']==selected['epoch'] and reconciliation['samples']==1429
        assert reconciliation['training_history_iou']==selected['val_iou'] and reconciliation['original_export_iou']==val['macro_iou']
        assert abs(reconciliation['history_residual'])<1e-12 and abs(reconciliation['export_residual'])<1e-12
        assert reconciliation['max_export_record_error']==0
        assert reconciliation['checkpoint_sha256_before']==reconciliation['checkpoint_sha256_after']
        assert abs(selection_delta-reconciliation['threshold_effect_export_minus_ge']-reconciliation['float32_effect_ge_minus_training'])<1e-12
    snapshots=[HERE/(label+'_'+p+'_snapshot.json') for p in ('final','first')]
    snapshot=next(read(p) for p in snapshots if p.exists() and read(p)['runtime']['phase']=='complete')
    runtime=snapshot['runtime'];assert runtime['training_rc']==runtime['validation_rc']==0 and snapshot['runtime_source_clean']
    assert runtime['source_git_commit']==source['source_git_commit'] and snapshot['inspection_number']<=2
    names=None
    for epoch in (20,40,60,80):
        d=read(folder/('diagnostics/epoch_'+str(epoch).zfill(3)+'.json'))
        assert d['epoch']==epoch and d['source_git_commit']==source['source_git_commit']
        assert d['state_sha256_before']==d['state_sha256_after'] and d['gradient_state_unchanged'] and d['rng_restored']
        current=[r['name'] for batch in d['batches'] for r in batch['cases']]
        assert len(current)==len(set(current))==32
        if names is None:names=current
        assert names==current
    for name in ('training.log','validation.log'):
        text=(folder/name).read_text(encoding='utf-8',errors='replace')
        assert not any(s in text for s in ('Traceback (most recent call last)','CUDA out of memory','RuntimeError:'))
    comparison['input_sha256']=dict(control=digest(baseline),candidate=digest(folder/'validation.json'))
    save(label+'_results/paired_comparison.json',comparison)
    proof=dict(verified=True,source_git_commit=source['source_git_commit'],seed=source['seed'],epochs=80,samples=1429,
        best_epoch=val['checkpoint_best_epoch'],actual_lr_matches_manifest=True,macro_metrics_recomputed_from_integer_counts=True,
        audits_state_preserving=True,test_split_accessed=False,inspection_number=snapshot['inspection_number'],
        seconds_after_training_end=snapshot['seconds_after_training_end'],within_30_minutes=snapshot['within_30_minutes'],
        export_minus_selection_iou=selection_delta,metric_reconciliation_verified=reconciliation is not None)
    save(label+'_results/independent_verification.json',proof)
    lines=[f'# {label} Val replication','',f"Source `{source['source_git_commit']}`; seed {source['seed']}; 80 epochs; Best {val['checkpoint_best_epoch']}.",'',
        '| Model | IoU | Dice | Precision | Recall | Brier |','|---|---:|---:|---:|---:|---:|']
    for name,data in (('R2 matched seed',base),('RS1',val)):
        lines.append('| '+name+' | '+' | '.join(f"{data['macro_'+m]*100:.4f}%" for m in METRICS[:-1])+f" | {data['macro_brier']:.6f} |")
    lines+=['',f"IoU delta {comparison['deltas']['iou']['mean']*100:+.4f} pp; grouped descriptive CI {[x*100 for x in comparison['deltas']['iou']['ci95']]} pp.",'',
        'This single result does not decide the two-seed replication. Complete both registered seeds regardless of the first numerical outcome. Precision is reported and is not an automatic veto. All metrics, small-area results and FP/FN counts are in paired_comparison.json.',
        '',f"Inspections {snapshot['inspection_number']}/2; seconds after training end {snapshot['seconds_after_training_end']:.3f}. Test was not accessed. HF upload requires its separate verified receipt."]
    for name in ('REPORT.md','README.md'):(folder/name).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return proof

def decision(deltas,ci):
    return dict(both_new_seeds_positive=bool(all(d>0 for d in deltas)),new_seed_mean_at_least_0_003=bool(np.mean(deltas)>=.003),
        grouped_descriptive_ci_positive=bool(ci[0]>0))

def aggregate():
    sources=read(HERE/'sources.json');pairs=[];names=None;new_deltas=[];all_results=[]
    for label,s in sources.items():
        assert read(HERE/(label+'_results/independent_verification.json'))['verified']
        base=read(DOCS/s['baseline_validation_relative']);candidate=read(HERE/(label+'_results/validation.json'))
        assert digest(DOCS/s['baseline_validation_relative'])==s['baseline_validation_sha256']
        result=compare(base,candidate);all_results.append(result);aa={r['name']:r for r in base['records']};bb={r['name']:r for r in candidate['records']}
        if names is None:names=sorted(aa)
        assert names==sorted(aa)==sorted(bb)
        pairs.append(np.array([bb[n]['iou']-aa[n]['iou'] for n in names]));new_deltas.append(result['deltas']['iou']['mean'])
    summary=interval(np.mean(pairs,axis=0),names);checks=decision(new_deltas,summary['ci95'])
    oldbase=DOCS/'repro_archive/20260908/recipe_execution/r2_results/validation.json'
    oldcandidate=DOCS/'repro_archive/20260911/regional_supervision_execution/rs1_results/validation.json'
    discovery=compare(read(oldbase),read(oldcandidate));all_results.insert(0,discovery)
    all_deltas=[r['deltas']['iou']['mean'] for r in all_results]
    result=dict(verified=True,protocol_sha256=digest(HERE/'PROTOCOL.md'),passed=all(checks.values()),checks=checks,
        new_seeds=[2027,3407],new_seed_iou_deltas=new_deltas,new_seed_mean_iou_delta=float(np.mean(new_deltas)),
        new_seed_delta_sample_sd=float(np.std(new_deltas,ddof=1)),new_seed_grouped_ci=summary,
        all_three_seed_mean_iou_delta=float(np.mean(all_deltas)),all_three_seed_delta_sample_sd=float(np.std(all_deltas,ddof=1)),
        all_three_seed_results=all_results,discovery_seed=1219,discovery_not_in_primary_gate=True,
        selected_common_recipe='RS1' if all(checks.values()) else 'R2',test_split_accessed=False,
        stable_test_gain_proven=False,second_text_innovation_proven=False)
    save('replication_summary.json',result)
    lines=['# IoU-first RS1 replication','', '| Seed | R2 Val IoU | RS1 Val IoU | Delta pp | Role |','|---|---:|---:|---:|---|']
    for r in all_results:lines.append(f"| {r['seed']} | {r['baseline']['iou']*100:.4f}% | {r['candidate']['iou']*100:.4f}% | {r['deltas']['iou']['mean']*100:+.4f} | {'discovery' if r['seed']==1219 else 'replication'} |")
    lines+=['',f"Registered two-new-seed criterion passed: {result['passed']}. Shared recipe for the next text experiment: {result['selected_common_recipe']}.",'',
        f"New-seed mean delta {result['new_seed_mean_iou_delta']*100:+.4f} pp; sample SD {result['new_seed_delta_sample_sd']*100:.4f} pp; descriptive grouped image CI {[v*100 for v in summary['ci95']]} pp.",'',
        'The grouped image interval is not evidence of across-training-seed statistical significance. The discovery seed does not decide the replication gate. Full negative results and secondary metrics remain in replication_summary.json. This stage has no new Test result or established second innovation.']
    (HERE/'REPLICATION_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return {k:v for k,v in result.items() if k!='all_three_seed_results'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',choices=('rs1s2027','rs1s3407'));p.add_argument('--aggregate',action='store_true');a=p.parse_args()
    if a.aggregate:print(json.dumps(aggregate(),indent=2))
    else:assert a.label;print(json.dumps(verify_completed(a.label),indent=2))

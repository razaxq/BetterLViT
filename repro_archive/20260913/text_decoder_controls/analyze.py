"""Matched decoder-control comparisons from immutable completed Val exports."""
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
    assert base['experiment'] in ('r2_single_cosine','t1_decoder_visual')
    assert candidate['experiment'] in ('t1_decoder_visual','t2_decoder_text')
    assert base['seed']==candidate['seed'] and a.keys()==b.keys()
    names=sorted(a);areas=np.array([a[n]['label_pixels'] for n in names]);assert np.array_equal(areas,[b[n]['label_pixels'] for n in names])
    small=areas<=np.quantile(areas,.25);deltas={}
    for m in METRICS:
        delta=np.array([b[n][m]-a[n][m] for n in names])
        deltas[m]=dict(mean=float(delta.mean()),small_mean=float(delta[small].mean()),**interval(delta,names))
    counts={}
    for label,rows in (('control',a),('candidate',b)):
        tp=sum(round(r['recall']*r['label_pixels']) for r in rows.values())
        counts[label]=dict(tp=tp,fp=sum(r['prediction_pixels'] for r in rows.values())-tp,
            fn=sum(r['label_pixels'] for r in rows.values())-tp)
    counts['delta']={k:counts['candidate'][k]-counts['control'][k] for k in ('tp','fp','fn')}
    return dict(seed=base['seed'],control_sha=base['checkpoint_git_commit'],candidate_sha=candidate['checkpoint_git_commit'],
        baseline={m:base['macro_'+m] for m in METRICS},candidate={m:candidate['macro_'+m] for m in METRICS},
        deltas=deltas,pixel_counts_reconstructed_and_metrics_verified=counts,
        small_area_cutoff=float(np.quantile(areas,.25)),small_count=int(small.sum()),split='validation',test_split_accessed=False,
        discovery_seed_only=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--label',choices=('t1','t2'),required=True);label=p.parse_args().label
    source=read(HERE/'sources.json')[label];folder=HERE/(label+'_results')
    val=read(folder/'validation.json');base=read(DOCS/'repro_archive/20260908/recipe_execution/r2_results/validation.json')
    assert base['checkpoint_git_commit']=='9eca26de5b301099805530edbf5a1a8718bea662'
    assert val['checkpoint_git_commit']==source['source_git_commit'] and val['seed']==source['seed']==1219
    mode={'t1':'visual','t2':'text'}[label]
    assert val['decoder_context']['mode']==mode and val['decoder_context']['parameters']==16896
    assert val['training_recipe']==base['training_recipe']
    comparison=compare(base,val)
    manifest=read(Path(source['local_repository'])/'experiment_manifests/active_decoder.json')
    history=read(folder/'epoch_history.json');assert [h['epoch'] for h in history]==list(range(1,81))
    assert np.allclose([h['lr'] for h in history],manifest['planned_epoch_lrs'],rtol=0,atol=1e-15)
    selected=max(history[5:],key=lambda h:h['val_iou']);assert selected['epoch']==val['checkpoint_best_epoch']
    selection_delta=val['macro_iou']-selected['val_iou']
    observations=[json.loads(line) for line in (folder/'decoder_observations.jsonl').read_text().splitlines()]
    assert [x['epoch'] for x in observations]==[1,10,40,80]
    for x in observations:
        assert x['source_git_commit']==source['source_git_commit']
        assert np.isfinite(list(x['gradients'].values())).all()
    runtime=read(folder/'runtime.json');assert runtime['training_rc']==runtime['validation_rc']==0 and runtime['phase']=='complete'
    snapshot=read(HERE/(label+'_final_snapshot.json'))
    assert snapshot['runtime_source_clean'] and snapshot['runtime']['source_git_commit']==source['source_git_commit']
    for name in ('training.log','validation.log'):
        log=(folder/name).read_text(encoding='utf-8',errors='replace')
        assert not any(s in log for s in ('Traceback (most recent last)','Traceback (most recent call last)','CUDA out of memory','RuntimeError:'))
    save(label+'_results/paired_comparison.json',comparison)
    if label=='t2':
        t1=read(HERE/'t1_results/validation.json');save('t2_results/t2_vs_t1.json',compare(t1,val))
    proof=dict(verified=True,source_git_commit=source['source_git_commit'],seed=1219,epochs=80,samples=1429,
        best_epoch=val['checkpoint_best_epoch'],actual_lr_matches_manifest=True,macro_metrics_recomputed_from_integer_counts=True,
        observations_recorded=True,test_split_accessed=False,inspection_number=snapshot['inspection_number'],
        seconds_after_training_end=snapshot['seconds_after_training_end'],within_30_minutes=snapshot['within_30_minutes'],
        export_minus_selection_iou=selection_delta,
        threshold_reconciliation_needed=abs(selection_delta)>=1e-7)
    save(label+'_results/independent_verification.json',proof)
    lines=[f'# {label.upper()} matched decoder control (validation only)','',
        f"Source `{source['source_git_commit']}`; seed1219;80epochs;Best{val['checkpoint_best_epoch']}.",'',
        '| Model | IoU | Dice | Precision | Recall |','|---|---:|---:|---:|---:|']
    for name,data in [('R2 reused',base),(label.upper(),val)]:
        lines.append('| '+name+' | '+' | '.join(f"{data['macro_'+m]*100:.4f}%" for m in METRICS[:4])+' |')
    lines.extend(['',f"IoU delta {comparison['deltas']['iou']['mean']*100:+.4f} pp; grouped descriptive CI {[x*100 for x in comparison['deltas']['iou']['ci95']]} pp.",'',
        'Single-seed discovery only; no Test access, stable gain or novelty claim. All small-area metrics and gradient observations are preserved.',
        f'Export-minus-checkpoint-selection IoU={selection_delta:.12g}. Training uses >=0.5 while frozen export uses >0.5; reconcile any material discrepancy before performance interpretation.'])
    lines.extend(['',f"Smallest-mask quartile: {comparison['small_count']} images, area <= {comparison['small_area_cutoff']:.0f} pixels.",
        '| Metric | Small-mask delta |','|---|---:|'])
    for m in METRICS[:4]:
        lines.append(f"| {m} | {comparison['deltas'][m]['small_mean']*100:+.4f} pp |")
    lines.extend(['',f"Brier: R2 {base['macro_brier']:.8f}; {label.upper()} {val['macro_brier']:.8f}; delta {comparison['deltas']['brier']['mean']:+.8f}.",
        f"Total pixel count differences (candidate minus control): {comparison['pixel_counts_reconstructed_and_metrics_verified']['delta']}. Counts do not replace per-image macro metrics.",
        '',f"Final inspection {snapshot['checked_sydney']}; {snapshot['seconds_after_training_end']:.3f} seconds after training ended; inspections {snapshot['inspection_number']}/2.",
        '', '| Observed Train epoch | Residual RMS / feature RMS | Attention entropy | Q gradient absmax |',
        '|---|---:|---:|---:|'])
    for o in observations:
        lines.append(f"| {o['epoch']} | {o['residual_rms']/o['feature_rms']:.6f} | {o['attention_entropy']:.6f} | {o['gradients']['decoder_context.query.weight']:.6g} |")
    lines.extend(['','These observations cover one ordinary Train batch at each of four epochs. Nonzero residuals/gradients show that the branch is active on those batches; they do not establish a beneficial causal effect or dataset-wide grounding.'])
    for name in ('REPORT.md','README.md'):(folder/name).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(proof,indent=2))

if __name__=='__main__':main()

"""Reuse bounded execution/archival primitives with explicit control-specific changes."""
from pathlib import Path
from remote_ops import HERE
old=HERE.parents[1]/'20260912/rs1_iou_replication'
def load(name):return (old/name).read_text(encoding='utf-8')
def put(name,value):(HERE/name).write_text(value,encoding='utf-8',newline='\n')
for name in ('inspect_run.py','publish.py','verify_downloads.py'):
    text=load(name).replace("('rs1s2027','rs1s3407')","('t1','t2')")
    if name=='inspect_run.py':
        text=text.replace("+(80-last['epoch'])*rate+120","+(80-last['epoch'])*rate")
        text=text.replace('remaining_audit_allowance_seconds=120','remaining_audit_allowance_seconds=0')
    put(name,text)
text=load('archive_completed.py').replace("('rs1s2027','rs1s3407')","('t1','t2')")
text=text.replace("    assert ck['regional_supervision']['mode']==runtime['manifest']['regional_mode']\n    assert ck['regional_supervision']['weight']==runtime['manifest']['regional_weight']",
    "    assert ck['decoder_context']['mode']==runtime['manifest']['decoder_context_mode']\n    assert ck['decoder_context']['parameters']==16896")
text=text.replace("regional_supervision=ck['regional_supervision']","decoder_context=ck['decoder_context']")
start=text.index('for epoch in (20,40,60,80):')
end=text.index("(run/'checkpoint_metadata.json')",start)
text=text[:start]+'''observations=[json.loads(line) for line in (run/'decoder_observations.jsonl').read_text().splitlines()]
assert [o['epoch'] for o in observations]==[1,10,40,80]
assert all(o['source_git_commit']==SOURCE['source_git_commit'] for o in observations)
''' + text[end:]
put('archive_completed.py',text)
text=load('upload_completed.py').replace("('rs1s2027','rs1s3407')","('t1','t2')")
text=text.replace('authorized completed visual-aux pilots','authorized completed decoder controls')
text=text.replace("'completed_validation_only_80e_independent_seed_replication'","'completed_validation_only_80e_decoder_control'")
text=text.replace("experiment_manifests/active_regional.json","experiment_manifests/active_decoder.json")
start=text.index("    if label in ('rs2','rs3'):")
end=text.index('    helper_b64=',start)
text=text[:start]+text[end:]
needle="    if (folder/'iou_reconciliation.json').exists():"
text=text.replace(needle,"    if label=='t2':\n        supplements['analysis/t2_vs_t1.json']=base64.b64encode((folder/'t2_vs_t1.json').read_bytes()).decode()\n"+needle)
put('upload_completed.py',text)
text=load('upload_verified_manifest.py').replace("'completed_validation_only_80e_independent_seed_replication'","'completed_validation_only_80e_decoder_control'")
put('upload_verified_manifest.py',text)
text=load('analyze.py')
text=text.replace('IoU-first paired replication analysis','Matched decoder-control comparison')
text=text[:text.index('\ndef verify_completed(')]
text=text.replace("    assert base['experiment']=='r2_single_cosine' and candidate['experiment']=='rs1_global_iou'", "    assert base['experiment'] in ('r2_single_cosine','t1_decoder_visual')\n    assert candidate['experiment'] in ('t1_decoder_visual','t2_decoder_text')")
text=text.replace("('r2',a),('rs1',b)","('control',a),('candidate',b)")
text=text.replace("counts['rs1'][k]-counts['r2'][k]","counts['candidate'][k]-counts['control'][k]")
text=text.replace('individual_seed_pass_is_not_overall_replication=True','discovery_seed_only=True')
text+='''
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
    for name in ('REPORT.md','README.md'):(folder/name).write_text('\\n'.join(lines)+'\\n',encoding='utf-8')
    print(json.dumps(proof,indent=2))

if __name__=='__main__':main()
'''
put('analyze.py',text)

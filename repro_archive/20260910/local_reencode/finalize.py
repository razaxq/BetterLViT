"""Independent CPU count checks and immutable completed-artifact archival."""
import json
from pathlib import Path
import shutil
import numpy as np
from analysis import digest, summarize, write_json

HERE=Path(__file__).resolve().parent
OUT=Path('D:/BetterLViT/outputs/local_reencode_20260910')


def main():
    state=json.loads((OUT/'state.json').read_text())
    source=OUT/'results'
    runtime=json.loads((source/'runtime.json').read_text())
    assert state['phase']==runtime['phase']=='complete'
    for name,meta in runtime['artifacts'].items():
        assert (source/name).stat().st_size==meta['bytes'] and digest(source/name)==meta['sha256']
    data=json.loads((source/'records.json').read_text())
    rows=data['records']
    training=json.loads((source/'training.json').read_text())
    audit=json.loads((source/'audit_records.json').read_text())['records']
    # Training stores audit mask filenames; the baseline loader returns image names.
    assert all(n.startswith('mask_') for n in training['audit_names'])
    audit_images=[n.replace('mask_','') for n in training['audit_names']]
    fit_names=set(training['fit_names']);audit_names=set(audit_images);val_names={r['name'] for r in rows}
    assert len(fit_names)==5460 and len(audit_names)==256 and len(val_names)==1429
    assert not fit_names&audit_names and not fit_names&val_names and not audit_names&val_names
    assert {r['name'] for r in audit}==audit_names
    import hashlib
    full_order=training['fit_names']+audit_images
    assert full_order==sorted(full_order,key=lambda n:hashlib.sha256(('local-reencode-v1:mask_'+n).encode()).hexdigest())
    assert [x['step'] for x in training['history']]==list(range(1,2401))
    assert abs(training['history'][0]['lr']-3e-4)<1e-14 and abs(training['history'][-1]['lr']-1e-6)<1e-14
    assert data['source_git_commit']==state['source_git_commit']==runtime['source_git_commit']
    assert data['baseline_state_unchanged'] and data['baseline_per_image_max_difference']==0 and not data['test_split_accessed']
    checked=0
    for row in rows+audit:
        assert row['selected_pixels']==10240
        assert 0<=row['selected_errors']<=row['error_pixels']
        assert 0<=row['errors_outside_old_logit_bound']<=row['error_pixels']
        for m in [row[k] for k in ('baseline','features','image')]+list(row['full_diagnostic'].values()):
            tp=m['label_pixels']-m['fn'];pp=tp+m['fp'];gt=m['label_pixels'];union=gt+m['fp']
            expected=dict(iou=tp/union if union else 1.,dice=2*tp/(pp+gt) if pp+gt else 1.,
                          precision=tp/pp if pp else 0.,recall=tp/gt if gt else 0.)
            assert tp>=0 and all(abs(m[k]-v)<1e-12 for k,v in expected.items())
            checked+=1
    summary=summarize(rows)
    assert summary==json.loads((source/'summary.json').read_text())
    valid=[r for r in rows if r['error_pixels']]
    diagnostic=dict(metric_rows_reconstructed=checked,
        fit_64_relative_loss_reduction={a:1-training['fit_64_loss_after'][a]/training['fit_64_loss_before'][a] for a in ('features','image')},
        audit_means={a:{k:float(np.mean([r[a][k] for r in audit])) for k in ('iou','dice','precision','recall','brier')} for a in ('baseline','features','image')},
        full_unmasked_diagnostic_iou={a:float(np.mean([r['full_diagnostic'][a]['iou'] for r in rows])) for a in ('features','image')},
        error_capture_macro=float(np.mean([r['selected_errors']/r['error_pixels'] for r in valid])),
        outside_old_bound_error_fraction_macro=float(np.mean([r['errors_outside_old_logit_bound']/r['error_pixels'] for r in valid])),
        outside_old_bound_error_fraction_pooled=sum(r['errors_outside_old_logit_bound'] for r in rows)/sum(r['error_pixels'] for r in rows))
    write_json(source/'diagnostics.json',diagnostic)
    proof=json.loads((source/'independent_verification.json').read_text())
    proof.update(independent_count_metric_rows=checked,split_names_disjoint=True,all_2400_steps_verified=True)
    write_json(source/'independent_verification.json',proof)
    target=HERE/'results';target.mkdir(exist_ok=True)
    for p in source.iterdir():
        if p.is_file():shutil.copyfile(p,target/p.name)
    for name in ('state.json','inspection_1.json'):shutil.copyfile(OUT/name,HERE/'execution'/name)
    print(json.dumps(dict(verified=True,metric_rows_reconstructed=checked,proceed=summary['proceed'],diagnostics=diagnostic)))


if __name__=='__main__':main()

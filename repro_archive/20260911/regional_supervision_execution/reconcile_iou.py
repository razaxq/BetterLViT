"""One bounded Val-only metric audit on a completed frozen Best checkpoint.

Retains the original exporter and training outputs. Separately evaluates the
inherited >= float32 training metric and > float64 exported metric on the same
probabilities, recording exact threshold ties and integer intersections/unions.
This does not inspect training progress or change checkpoint selection.
"""
import argparse
import hashlib
import json
from remote_ops import HERE, environment, remote, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', choices=('rs1', 'rs2', 'rs3'), required=True)
    label = parser.parse_args().label
    source = json.loads((HERE/'sources.json').read_text())[label]
    folder = HERE/(label+'_results')
    assert json.loads((folder/'runtime.json').read_text())['phase'] == 'complete'
    assert not (folder/'iou_reconciliation.json').exists(), 'Audit already exists'
    result = remote('SOURCE='+repr(source)+'\nENV='+repr(environment(source['source_git_commit'], source['profile']))+'\n'+'''
import contextlib, hashlib, json, os, sys
from pathlib import Path
import numpy as np
os.environ.update(ENV)
os.chdir(SOURCE['repository']);sys.path.insert(0, SOURCE['repository'])
with contextlib.redirect_stdout(sys.stderr):
    import torch
    from tools import export_validation_metrics as exporter
    from utils import iou_on_batch_gpu
    config = exporter.config
    run = Path(SOURCE['remote_run'])
    runtime = json.loads((run/'runtime.json').read_text())
    assert runtime['phase']=='complete' and runtime['training_rc']==runtime['validation_rc']==0
    original = json.loads((run/'validation.json').read_text())
    reference = {r['name']:r for r in original['records']}
    history = json.loads((run/'epoch_history.json').read_text())
    def digest(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024), b''):h.update(block)
        return h.hexdigest()
    checkpoint_path=Path(runtime['best_checkpoint'])
    before=digest(checkpoint_path)
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    assert checkpoint['source_git_commit']==exporter.git_commit()==SOURCE['source_git_commit']
    assert checkpoint['best_epoch']==original['checkpoint_best_epoch']
    torch.backends.cudnn.enabled=config.cudnn_enabled
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=config.deterministic_training
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(config.deterministic_training)
    model=exporter.build_model()
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    model=model.cuda().eval()
    _,loader=exporter.validation_loader(16)
    rows=[];weighted=0.0
    with torch.inference_mode():
        for batch,names in loader:
            pred=model(batch['image'].cuda(non_blocking=True),batch['input_ids'].cuda(non_blocking=True),batch['attention_mask'].cuda(non_blocking=True))
            masks=batch['label'].cuda(non_blocking=True)
            weighted+=len(names)*float(iou_on_batch_gpu(masks,pred))
            probabilities=pred[:,0].float().cpu();labels=batch['label'].bool()
            for index,name in enumerate(names):
                p=probabilities[index];gt=labels[index];r=dict(name=name)
                for key,binary in (('gt',p>0.5),('ge',p>=0.5)):
                    r[key+'_intersection']=int((binary&gt).sum())
                    r[key+'_union']=int((binary|gt).sum())
                    r[key+'_iou']=r[key+'_intersection']/max(r[key+'_union'],1)
                r['threshold_ties']=int((p==0.5).sum())
                r['threshold_ties_positive']=int(((p==0.5)&gt).sum())
                r['export_record_abs_error']=abs(r['gt_iou']-reference[name]['iou'])
                rows.append(r)
    training_recomputed=weighted/len(rows)
    export_recomputed=float(np.mean([r['gt_iou'] for r in rows]))
    ge_double=float(np.mean([r['ge_iou'] for r in rows]))
    selected=max(history[5:],key=lambda r:r['val_iou'])
    result=dict(source_git_commit=SOURCE['source_git_commit'],test_split_accessed=False,
        samples=len(rows),checkpoint_best_epoch=checkpoint['best_epoch'],checkpoint_sha256_before=before,
        checkpoint_sha256_after=digest(checkpoint_path),original_export_sha256=digest(run/'validation.json'),
        training_history_iou=selected['val_iou'],training_ge_float32_recomputed=training_recomputed,
        original_export_iou=original['macro_iou'],export_gt_float64_recomputed=export_recomputed,
        ge_float64_recomputed=ge_double,
        threshold_effect_export_minus_ge=export_recomputed-ge_double,
        float32_effect_ge_minus_training=ge_double-training_recomputed,
        history_residual=training_recomputed-selected['val_iou'],
        export_residual=export_recomputed-original['macro_iou'],
        max_export_record_error=max(r['export_record_abs_error'] for r in rows),
        exact_half_pixels=sum(r['threshold_ties'] for r in rows),
        exact_half_positive_pixels=sum(r['threshold_ties_positive'] for r in rows),records=rows)
    result['verified']=(len(rows)==1429 and len({r['name'] for r in rows})==1429
        and result['checkpoint_sha256_before']==result['checkpoint_sha256_after']
        and result['max_export_record_error']==0 and abs(result['history_residual'])<1e-12
        and abs(result['export_residual'])<1e-12)
print(json.dumps(result))
''', timeout=300)
    result['audit_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save(label+'_results/iou_reconciliation.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}, indent=2))
    assert result['verified'], 'Metric discrepancy remains unexplained; original results preserved'


if __name__ == '__main__':
    from pathlib import Path
    main()

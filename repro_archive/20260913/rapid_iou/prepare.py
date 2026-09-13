"""Freeze a bounded inference-only screen before any new predictions are read."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = HERE.parents[2]
old = json.loads((DOCS/'repro_archive/20260910/recipe_test/test_plan.json').read_text())
sources = [s for s in old['arms'] if s['label'] == 'r2s1219']
assert len(sources) == 1
refs = [DOCS/'repro_archive/20260908/recipe_execution/r2_results/validation.json']
(HERE/'references').mkdir(parents=True, exist_ok=True)
for s, path in zip(sources, refs):
    data = json.loads(path.read_text())
    assert data['checkpoint_git_commit'] == s['source_git_commit']
    assert data['seed'] == s['seed'] and data['checkpoint_best_epoch'] == s['best_epoch']
    s['validation_reference_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (HERE/'references'/(s['label']+'_validation.json')).write_bytes(path.read_bytes())
    s.pop('validation_source', None)
    s.pop('validation_sha256', None)
plan = dict(version=1, sources=sources, batch_size=16, image_size=224,
    threshold_grid=[round(.3 + .01*i, 2) for i in range(41)],
    threshold_rule='float32 probability strictly greater than float32 threshold',
    threshold_selection='maximum Val per-image macro IoU; exact ties closest to 0.5 then lower',
    baseline='r2s1219 at 0.5', candidates=['r2s1219_calibrated'],
    ensemble_allowed=False, no_training=True, no_tta=True, no_component_filter=True,
    new_architecture_claim=False, validation_samples=1429, test_samples=2113,
    test_gate='Val macro IoU improvement >=0.001 and smallest-mask-quartile IoU >= baseline',
    test_selection='Freeze one threshold per accepted inference method; no Test search or winner reselection',
    historical_test_access=True, final_reporting='Test macro IoU/Dice; report all frozen methods including failures',
    validation_claim='Discovery only; threshold selection reuses Val; bootstrap is descriptive',
    storage='No probability-map disk cache; only small count/provenance exports; preserve all models',
    user_priority='Quick IoU improvement with one existing model, no additional parameters or forward passes')
target = HERE/'plan.json'
assert not target.exists(), 'Do not overwrite a frozen plan'
target.write_text(json.dumps(plan, indent=2)+'\n', encoding='utf-8', newline='\n')
print(json.dumps(dict(prepared=True, sources=[s['source_git_commit'] for s in sources])))

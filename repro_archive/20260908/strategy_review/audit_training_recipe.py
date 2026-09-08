"""Recompute restart-associated changes from archived P12 history, offline."""
import hashlib
import json
import subprocess
from pathlib import Path

base = Path(__file__).resolve().parent
source = Path('D:/BetterLViT/race_pe_c4_work')
sha = subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
assert sha == 'add4908a0d6f702b0a10c4581725b535543829b8'
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=source,text=True).strip()
history_path = base.parent/'visual_prior_execution/p12_results/epoch_history_rounded.json'
history = {r['epoch']:r for r in json.loads(history_path.read_text(encoding='utf-8'))}
loader = (source/'Load_Dataset.py').read_text(encoding='utf-8')
trainer = (source/'train_model.py').read_text(encoding='utf-8')
assert 'np.rot90(image, k, axes=(0, 1))' in loader
assert 'image = np.flip(image, axis=axis).copy()' in loader
assert "'input_ids': sample['input_ids']" in loader
assert 'CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)' in trainer
rows = [{'epoch':e,'previous_lr':history[e-1]['lr'],'lr':history[e]['lr'],
         'previous_val_iou':history[e-1]['val_iou'],'val_iou':history[e]['val_iou'],
         'delta_val_iou':round(history[e]['val_iou']-history[e-1]['val_iou'],4)}
        for e in (11,21,31,41,51,61,71)]
result = {'source_git_commit':sha,'p12_source_git_commit':'5d09913d46863073cba93159af4ed61f88fdd96e',
    'test_split_accessed':False,'new_training':False,
    'history_sha256':hashlib.sha256(history_path.read_bytes()).hexdigest(),
    'augmentation':{'rot90_then_flip_branch_probability':.5,
        'small_rotation_branch_probability':.25,'unchanged_branch_probability':.25,
        'text_tokens_unchanged':True,'race_zone_basis_transformed':True,
        'interpretation':'Orientation distribution and spatial text compatibility deserve controlled testing; transformed region bases are not being claimed misaligned.'},
    'restart_boundaries':rows,
    'all_seven_boundaries_have_lower_val_iou':all(r['delta_val_iou']<0 for r in rows),
    'causal_limitation':'A transient validation drop after a scheduled restart is not proof that restarts reduce the best achievable result.',
    'decision':'Two separate C4 recipe comparisons: orientation-preserving augmentation, then a single learning-rate decay; no new module, supervision, or Test selection.'}
(base/'training_recipe_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))

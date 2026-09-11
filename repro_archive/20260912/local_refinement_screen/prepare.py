"""Register F's five-fold Train-only interface screen, before model fitting."""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
B=HERE.parent.parent/'20260911/text_head_screening'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value):(HERE/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')
original=json.loads((B/'split.json').read_text())['records']
rows=[]
for r in original:
    if r['partition']!='fit':continue
    rows.append(dict(cache_row=len(rows),index=r['index'],name=r['name'],mask_name=r['mask_name'],
        group_id=r['group_id'],explicit_patient_id=r['explicit_patient_id'],
        fold=int(hashlib.sha256(('local-f-v1:'+r['group_id']).encode()).hexdigest(),16)%5,
        source_text=r['source_text'],eligible=r['eligible'],semantic_group=r['semantic_group']))
assert len(rows)==4585 and len({r['index'] for r in rows})==4585
for g in {r['group_id'] for r in rows}:assert len({r['fold'] for r in rows if r['group_id']==g})==1
write('split.json',dict(kind='fivefold_head_crossfit_original_B_fit_only',records=rows))
for name in ('analysis.py','heads.py'):
    target='mass_projection.py' if name=='heads.py' else name
    (HERE/target).write_bytes((B/name).read_bytes())
base=json.loads((B/'manifest.json').read_text())
manifest={k:base[k] for k in ('baseline_repository','baseline_source_git_commit','baseline_checkpoint',
    'baseline_checkpoint_sha256','baseline_checkpoint_epoch_zero_based','train_workbook_sha256','train_membership_sha256','train_count')}
manifest.update(phase='F_frozen_visual_interface_fivefold_crossfit',seed=1219,folds=5,
    split_sha256=digest(HERE/'split.json'),original_b_split_sha256=digest(B/'split.json'),
    b_directory='/root/text_head_b_49905dbb',b_cache_manifest_sha256=digest(B/'results/cache_manifest.json'),
    inherited_projection_sha256=digest(HERE/'mass_projection.py'),
    sample_count=4585,fold_counts={str(f):sum(r['fold']==f for r in rows) for f in range(5)},
    explicit_patient_id_samples=sum(r['explicit_patient_id'] for r in rows),
    variants=['coarse_free','coarse_mass','fine_free','fine_mass'],
    candidate_points=1024,candidate_selection='descending sigmoid(z)*(1-sigmoid(z)), stable pixel-index tie break',
    feature_cache_shape=[4585,1024,64],feature_cache_dtype='float16',
    point_index_shape=[4585,1024],point_index_dtype='uint16',
    exact_cache_bytes=4585*1024*(64*2+2),minimum_free_bytes=256000000,
    batch_size=16,steps_per_fold=2048,optimizer='AdamW',lr_start=.001,lr_end=.00001,
    weight_decay=.0001,gradient_norm_clip=5.,hidden_channels=[128,64],residual_bound=.5,
    loss='Original R2 WeightedDiceFocal; .5 Dice + .5 focal, gamma2',
    projection_scope='selected1024 only; all other probabilities unchanged',mass_error_tolerance=.02,
    threshold=.5,threshold_operator='>',training_checkpoints=[512,1024,1536,2048],
    bootstrap_replicates=10000,bootstrap_seed=1219,
    gate=dict(minimum_oof_iou_gain=.001,group_ci_lower_positive=True,minimum_positive_folds=4,
        dice_no_drop=True,precision_no_drop=True,small_dice_no_drop=True,small_recall_no_drop=True,brier_no_increase=True),
    fine_interface_gate='fine candidate must pass base gate and beat matching coarse candidate with group CI lower>0',
    automatic_full_training=False,automatic_text_training=False,official_validation_accessed=False,
    test_split_accessed=False,old_b_holdout_accessed=False,shared_fs_written=False,
    maximum_inspections=2,initial_check_completion_margin_seconds=180)
write('manifest.json',manifest)
print(json.dumps(dict(samples=4585,folds=manifest['fold_counts'],cache_bytes=manifest['exact_cache_bytes'])))

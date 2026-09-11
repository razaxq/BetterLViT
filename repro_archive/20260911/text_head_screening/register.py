"""Freeze Train membership/text split before any new-head optimization."""
import argparse
import json
from pathlib import Path
from openpyxl import load_workbook
from analysis import digest,write_json
from split_policy import records,counts

HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--train-folder',type=Path,default=Path('D:/BetterLViT/BetterLViT/BetterLViT_完整迁移包_2026-08-16/03_数据/datasets/Covid19/Train_Folder'))
TRAIN=p.parse_args().train_folder
old=json.loads((HERE.parent/'text_grounding_execution_v3/manifest.json').read_text())
names=sorted(p.name for p in (TRAIN/'labelcol').iterdir() if p.is_file())
texts={str(row[0]):str(row[1]) for row in load_workbook(TRAIN/'Train_Val_text.xlsx',read_only=True,data_only=True).active.values if row[0] in set(names)}
assert digest(TRAIN/'Train_Val_text.xlsx')==old['train_workbook_sha256']
rows=records(names,texts)
write_json(HERE/'split.json',dict(kind='head_only_internal_holdout_not_independent_of_R2_training',records=rows,counts=counts(rows)))
keys=['baseline_repository','baseline_source_git_commit','baseline_checkpoint','baseline_best_epoch',
      'baseline_checkpoint_epoch_zero_based','baseline_checkpoint_sha256','train_membership_sha256',
      'train_workbook_sha256','train_count','registered_text_policy_sha256']
manifest={k:old[k] for k in keys}
manifest.update(phase='B_frozen_R2_matched_residual_head_screening',version=1,seed=1219,
    batch_size=16,steps=1024,cache_batch_size=16,lr_start=.001,lr_end=.00001,weight_decay=.0001,
    optimizer='Adam',loss='R2 WeightedDiceFocal with exact default .5/.5 weights and gamma2',
    variants=['T1','T2','T3','T4','image','template'],projected=['T3','T4','image','template'],
    fit_only_eligible=True,eligibility='unilateral_or_asymmetric_bilateral_for_all_six_heads',
    official_validation_accessed=False,test_split_allowed=False,baseline_updates=False,
    split_sha256=digest(HERE/'split.json'),split_counts=counts(rows),
    feature_shape=[len(rows),64,28,28],feature_dtype='float16',
    logits_shape=[len(rows),1,224,224],logits_dtype='float32',
    mask_shape=[len(rows),6272],mask_dtype='uint8_packed_little',
    raw_cache_bytes=len(rows)*(64*28*28*2+224*224*4+6272),
    cache_budget_bytes=1_900_000_000,minimum_free_after_cache_bytes=1_000_000_000,
    maximum_mass_error_pixels=.02,residual_logit_bound=.5,maximum_inspections=2,
    first_check_delay_seconds=480,completion_check_margin_seconds=300,
    final_holdout_once=True,training_telemetry_steps=[0,256,512,1024],
    mechanism_screen_gate=dict(minimum_eligible_delta_iou=.001,paired_group_ci_lower_gt=0.,
        minimum_incremental_delta_iou=0.,dice_no_drop=True,precision_no_drop=True,
        small_dice_no_drop=True,small_recall_no_drop=True,brier_no_increase=True),
    no_automatic_full_training=True,
    limitations=['R2 already trained on all original Train samples',
        'Image control removes text only from the new branch; frozen R2 features contain text',
        'Hash groups without explicit sub-S patient ID fall back to image names',
        'Same parameter shapes are a capacity control, not identical statistical capacity',
        'Single frozen feature layer, no augmentation, no final Test claim'])
assert manifest['raw_cache_bytes']<manifest['cache_budget_bytes']
write_json(HERE/'manifest.json',manifest)
print(json.dumps(dict(split=counts(rows),raw_cache_bytes=manifest['raw_cache_bytes'],manifest_sha256=digest(HERE/'manifest.json'))))

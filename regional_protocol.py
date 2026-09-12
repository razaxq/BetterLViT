"""Immutable RS1/RS2/RS3 launch contract."""
from training_recipe import rates_equal, planned_rates

PROFILES = {'rs1_global_iou':'global', 'rs2_local_iou':'local', 'rs3_balanced_iou':'balanced'}


def validate_regional_manifest(m):
    assert m['profile'] in PROFILES and m['regional_mode'] == PROFILES[m['profile']]
    assert m['epochs'] == 80 and m['seed'] in (1219, 2027, 3407) and m['batch_size'] == 16 and m['image_size'] == 224
    assert m['train_drop_last'] and m['num_workers'] == 4
    assert m['selection_metric'] == 'iou' and m['threshold'] == .5
    assert m['loss_name'] == 'dice_focal' and m['initialization'] == 'from_scratch'
    assert m['augmentation_policy'] == 'legacy' and m['lr_schedule'] == 'single_cosine'
    assert m['optimizer'] == 'Adam' and m['weight_decay'] == 1e-4
    assert not any(m[k] for k in ('test_split_allowed','auto_test_evaluate','lora','boundary_loss','visual_prior'))
    assert m['new_supervision'] and m['auto_validation_export']
    assert m['window'] == 56 and m['stride'] == 28 and m['eps'] == 1e-6
    assert 0 < m['regional_weight'] <= .25 and m['calibration_frozen']
    assert m['audit_epochs'] == [20,40,60,80]
    assert rates_equal(m['planned_epoch_lrs'], planned_rates('single_cosine'))

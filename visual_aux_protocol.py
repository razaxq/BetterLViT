"""Frozen S1/S2 manifest contract, separate from historical R2 recipes."""
from training_recipe import planned_rates, rates_equal

PROFILES={'s1_r2_pixel_aux':'pixel','s2_r2_visual_aux':'regional'}


def validate_visual_aux_manifest(m):
    assert m['visual_aux_mode']==PROFILES[m['profile']]
    assert m['epochs']==80 and m['batch_size']==16 and m['seed'] in (1219,2027,3407)
    assert m['initialization']=='from_scratch' and m['selection_metric']=='iou' and m['threshold']==.5
    assert m['augmentation_policy']=='legacy' and m['lr_schedule']=='single_cosine'
    assert m['loss_name']=='dice_focal' and m['new_supervision']
    assert not any(m[k] for k in ('test_split_allowed','auto_test_evaluate','lora','boundary_loss','visual_prior','text_auxiliary','inference_routing'))
    expected={'pixel':.02,'presence':.01 if m['visual_aux_mode']=='regional' else 0.,
              'occupancy':.005 if m['visual_aux_mode']=='regional' else 0.}
    assert m['effective_aux_coefficients']==expected
    assert rates_equal(m['planned_epoch_lrs'],planned_rates('single_cosine'))
    assert m['train_audit_epochs']==[20,40,60,80]

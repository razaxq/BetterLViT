"""Strict manifest checks for the matched P8 restart, separate from R2 recipes."""
from paper_experiments import get_paper_experiment
from training_recipe import validate_recipe_manifest

def validate_p8_manifest(manifest):
    assert manifest['profile'] in ('p8_r2_original','p8_r2_binding')
    profile=get_paper_experiment(manifest['profile'])
    assert manifest['race_enabled'] and not manifest['race_pe_enabled']
    assert manifest['race_aux_weight']==profile['race_aux_weight']==.05
    assert manifest['race_binding_repair']==profile['race_binding_repair']
    assert manifest['new_supervision'] is True
    assert manifest['unknown_report_policy']=='legacy_fallback'
    assert manifest['count_target_policy']=='original_p8_unchanged'
    assert manifest['seed']==1219 and manifest['num_workers']==4
    assert manifest['auto_validation_export'] and manifest['train_drop_last']
    # Reuse numerical recipe checks while explicitly validating added P8 supervision above.
    recipe=dict(manifest,profile='r2_single_cosine',new_supervision=False)
    validate_recipe_manifest(recipe)

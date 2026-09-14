"""Pinned auxiliary-only control; automatic fixed-Best Test follows user preference."""
from paper_experiments import get_paper_experiment
from training_recipe import validate_recipe_manifest
def validate_manifest(m):
    assert m['profile']=='p8_r2_binding_aux' and m['paper_id']=='P8BA'
    p=get_paper_experiment(m['profile'])
    for key,value in dict(race_enabled=True,race_pe_enabled=False,race_route_enabled=False,
                          race_binding_repair=True,race_aux_weight=.05).items():
        assert m[key]==p[key]==value,key
    assert m['seed']==1219 and m['num_workers']==4 and m['train_drop_last']
    assert m['auto_validation_export'] and m['auto_test_evaluate'] and m['test_split_allowed']
    assert m['unknown_report_policy']=='legacy_fallback' and m['count_target_policy']=='original_p8_unchanged'
    assert m['control_source_git_commit']=='c6e52a9b13576cab9e2451c550d03ad274bc09e2'
    validate_recipe_manifest(dict(m,profile='r2_single_cosine',new_supervision=False,
                                  auto_test_evaluate=False,test_split_allowed=False))

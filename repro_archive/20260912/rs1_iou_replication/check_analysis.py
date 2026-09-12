"""Check consequential analysis behavior against historical exports and synthetic cases."""
import copy,json
import numpy as np
from analyze import compare,validate_export,interval,decision
from remote_ops import HERE,DOCS,read,save

def main():
    base=read(DOCS/'repro_archive/20260908/recipe_execution/r2_results/validation.json')
    candidate=read(DOCS/'repro_archive/20260911/regional_supervision_execution/rs1_results/validation.json')
    value=compare(base,candidate)
    assert abs(value['deltas']['iou']['mean']-(candidate['macro_iou']-base['macro_iou']))<1e-12
    assert value['deltas']['precision']['mean']<0
    assert all(decision([.004,.004],[.002,.006]).values()),'Secondary precision drop must not veto IoU criterion'
    assert not all(decision([.008,-.001],[.001,.006]).values()),'Mean gain must not hide a negative new seed'
    assert not all(decision([.002,.002],[.001,.003]).values()),'Do not lower registered practical effect'
    assert not all(decision([.004,.004],[-.001,.008]).values()),'Keep the registered grouped interval check'
    check=interval([.2,.2,.2],['sub-S1_a','sub-S1_b','covid_1'])
    assert check['groups']==2 and check['explicit_patient_groups']==1 and np.allclose(check['ci95'],[.2,.2])
    bad=copy.deepcopy(candidate);bad['records'][0]['prediction_pixels']+=100
    try:validate_export(bad)
    except AssertionError:pass
    else:raise AssertionError('Corrupt pixel counts were accepted')
    proof=dict(verified=True,historical_metrics_and_integer_counts=True,negative_seed_rejected=True,
        precision_is_not_a_veto=True,minimum_effect_preserved=True,group_bootstrap_checked=True,
        corrupt_count_rejected=True,formal_training_performed=False,test_split_accessed=False)
    save('analysis_checks.json',proof);print(json.dumps(proof))

if __name__=='__main__':main()

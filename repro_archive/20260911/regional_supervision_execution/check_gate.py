"""Verify absolute and incremental gates before candidate results exist."""
import copy,json,sys
from pathlib import Path
from remote_ops import HERE,save
sys.path.insert(0,'D:/BetterLViT/regional_rs1_work/tools')
from compare_regional import compare


def fixture(profile,delta=0):
    vals=dict(iou=.72+delta,dice=.82+delta,precision=.82+delta,recall=.86+delta,brier=.02-delta)
    rows=[dict(name=str(i),label_pixels=i+1,**vals) for i in range(1429)]
    return dict(experiment=profile,split='validation',test_split_accessed=False,checkpoint_git_commit='a'*40,
        analysis_git_commit='a'*40,epochs=80,samples=1429,threshold=.5,selection_metric='iou',
        text_use_lora=False,boundary_loss_weight=0,training_recipe=dict(lr_schedule='single_cosine',augmentation_policy='legacy'),
        seed=1219,records=rows,**{'macro_'+k:v for k,v in vals.items()})


if __name__=='__main__':
    base=fixture('r2_single_cosine')
    assert compare(base,fixture('rs1_global_iou',.004))['passed']
    assert not compare(base,fixture('rs1_global_iou',.002))['passed']
    regional=compare(fixture('rs1_global_iou'),fixture('rs2_local_iou',.002),True)
    assert regional['passed'] and set(regional['checks'])=={'iou_minimum','iou_ci_positive'}
    invalid=fixture('rs1_global_iou',.004);invalid['test_split_accessed']=True
    try:compare(base,invalid)
    except AssertionError:pass
    else:raise AssertionError('Test data allowed')
    invalid=fixture('rs1_global_iou',.004);invalid['records'][0]['label_pixels']+=1
    try:compare(base,invalid)
    except AssertionError:pass
    else:raise AssertionError('Unmatched labels allowed')
    result=dict(status='ok',checks=['absolute_pass','absolute_threshold_fail','incremental_threshold',
        'test_access_rejected','matched_ground_truth_required'])
    save('preflight/gate_checks.json',result);print(json.dumps(result))

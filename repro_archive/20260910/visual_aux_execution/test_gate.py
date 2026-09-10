"""Verify fixed advancement gates and split isolation on synthetic records."""
import copy
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,'D:/BetterLViT/visual_aux_work/tools')
from compare_visual_aux import compare


def fixture(delta=.004):
    a=dict(split='validation',test_split_accessed=False,checkpoint_git_commit='a'*40,analysis_git_commit='a'*40,
        epochs=80,samples=1429,threshold=.5,selection_metric='iou',text_use_lora=False,boundary_loss_weight=0.,
        training_recipe=dict(lr_schedule='single_cosine',augmentation_policy='legacy'),seed=1219,experiment='r2_single_cosine',
        records=[dict(name=str(i),label_pixels=i+1,iou=.7,dice=.8,precision=.8,recall=.8,brier=.02) for i in range(1429)],
        macro_iou=.7,macro_dice=.8,macro_precision=.8,macro_recall=.8,macro_brier=.02)
    b=copy.deepcopy(a);b.update(experiment='s1_r2_pixel_aux',checkpoint_git_commit='b'*40,analysis_git_commit='b'*40,macro_iou=.7+delta)
    for r in b['records']:r['iou']+=delta
    return a,b


class GateTests(unittest.TestCase):
    def test_zero_gain_stops(self):self.assertFalse(compare(*fixture(0))['passed'])
    def test_registered_gain_passes(self):self.assertTrue(compare(*fixture())['passed'])
    def test_test_access_rejected(self):
        a,b=fixture();b['split']='test';b['test_split_accessed']=True
        with self.assertRaises(AssertionError):compare(a,b)
    def test_brier_regression_stops(self):
        a,b=fixture();b['macro_brier']=.021
        for r in b['records']:r['brier']=.021
        self.assertFalse(compare(a,b)['passed'])
    def test_provenance_mismatch_rejected(self):
        a,b=fixture();b['analysis_git_commit']='c'*40
        with self.assertRaises(AssertionError):compare(a,b)
    def test_regional_increment_distinct_threshold(self):
        a,b=fixture(.002);a['experiment']='s1_r2_pixel_aux';b['experiment']='s2_r2_visual_aux'
        self.assertTrue(compare(a,b,True)['passed'])
        a['experiment']='r2_single_cosine'
        self.assertFalse(compare(a,b)['passed'])


if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GateTests))
    (Path(__file__).resolve().parent/'preflight/gate_tests.json').write_text(json.dumps(
        dict(status='ok' if r.wasSuccessful() else 'failed',tests=r.testsRun,errors=r.errors,failures=r.failures),indent=2)+'\n')
    sys.exit(0 if r.wasSuccessful() else 1)

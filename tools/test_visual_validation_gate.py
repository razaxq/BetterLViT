import copy
import unittest
from compare_visual_validation import compare


def fixture(delta=0):
    records = [{'name':str(i),'label_pixels':i+1,'iou':.6+delta,'dice':.7+delta,
                'precision':.8,'recall':.8} for i in range(1429)]
    return dict(split='validation',test_split_accessed=False,checkpoint_git_commit='a'*40,
        analysis_git_commit='a'*40,selection_metric='iou',threshold=.5,epochs=80,samples=1429,
        text_use_lora=False,boundary_loss_weight=0,seed=1219,experiment='c4_race_pe_control',
        records=records,macro_iou=.6+delta,macro_dice=.7+delta,macro_precision=.8,macro_recall=.8)


class GateTests(unittest.TestCase):
    def test_registered_effect_and_null(self):
        self.assertTrue(compare(fixture(),fixture(.004))['passed'])
        self.assertFalse(compare(fixture(),fixture())['passed'])

    def test_test_and_wrong_provenance_are_rejected(self):
        for change in ({'split':'test'}, {'test_split_accessed':True},
                       {'analysis_git_commit':'b'*40}, {'epochs':20}, {'seed':3407}):
            candidate=fixture(.004)
            candidate.update(change)
            with self.assertRaises(ValueError): compare(fixture(),candidate)

    def test_record_mismatch_and_forged_summary_are_rejected(self):
        for mode in ('area','duplicate','nan','summary'):
            candidate=copy.deepcopy(fixture(.004))
            if mode=='area': candidate['records'][0]['label_pixels']=999
            if mode=='duplicate': candidate['records'][0]['name']='1'
            if mode=='nan': candidate['records'][0]['iou']=float('nan')
            if mode=='summary': candidate['macro_iou']=.9
            with self.assertRaises(ValueError): compare(fixture(),candidate)


if __name__=='__main__':
    unittest.main()

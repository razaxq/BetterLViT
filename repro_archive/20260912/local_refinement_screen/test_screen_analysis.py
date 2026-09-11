"""Regression checks for the stop/go decisions controlling further GPU work."""
import copy
import unittest
from screen_analysis import gate

class GateTests(unittest.TestCase):
    def setUp(self):
        self.t=dict(minimum_oof_iou_gain=.001,minimum_positive_folds=4)
        self.good=dict(delta_iou=.002,group_ci95=[.001,.003],delta=dict(dice=.001,precision=.001,brier=-.001),
            small_delta=dict(dice=.001,recall=.001),fold_delta_iou={str(f):.002 for f in range(5)})
    def test_all_required(self):self.assertTrue(gate(self.good,self.t)['passed'])
    def test_tiny_gain_does_not_pass(self):
        self.good['delta_iou']=.00099;self.assertFalse(gate(self.good,self.t)['passed'])
    def test_ci_crossing_zero_does_not_pass(self):
        self.good['group_ci95'][0]=0;self.assertFalse(gate(self.good,self.t)['passed'])
    def test_precision_tradeoff_does_not_pass(self):
        self.good['delta']['precision']=-1e-8;self.assertFalse(gate(self.good,self.t)['passed'])
    def test_small_lesion_tradeoff_does_not_pass(self):
        self.good['small_delta']['recall']=-1e-8;self.assertFalse(gate(self.good,self.t)['passed'])
    def test_only_three_positive_folds_does_not_pass(self):
        self.good['fold_delta_iou']['3']=0;self.good['fold_delta_iou']['4']=-.001;self.assertFalse(gate(self.good,self.t)['passed'])

if __name__=='__main__':unittest.main()

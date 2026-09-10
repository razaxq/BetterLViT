"""Regression checks for threshold semantics and no post-hoc progression."""
import copy
import unittest
import numpy as np
from analysis import metrics, summarize


class Checks(unittest.TestCase):
    def test_strict_threshold(self):
        r=metrics(np.array([.5,.50001,0.]),np.array([0,1,0]))
        self.assertEqual((r['iou'],r['dice'],r['fp'],r['fn']),(1.,1.,0,0))

    def test_empty_masks(self):
        r=metrics(np.zeros(4),np.zeros(4))
        self.assertEqual((r['iou'],r['dice'],r['precision'],r['recall']),(1.,1.,0.,0.))

    def rows(self):
        base=dict(iou=.70,dice=.80,precision=.82,recall=.82,brier=.05,fp=8,fn=8,label_pixels=50)
        return [dict(name=str(i),baseline=copy.deepcopy(base),features=copy.deepcopy(base),image=copy.deepcopy(base)) for i in range(16)]

    def test_identical_fails(self):self.assertFalse(summarize(self.rows())['proceed'])

    def test_ablation_matters(self):
        rows=self.rows()
        for r in rows:r['image']['iou']+=.004;r['features']['iou']+=.004
        self.assertFalse(summarize(rows)['proceed'])

    def test_safety_gates(self):
        rows=self.rows()
        for r in rows:r['image']['iou']+=.004
        self.assertTrue(summarize(rows)['proceed'])
        rows[0]['image']['recall']-=.001
        self.assertFalse(summarize(rows)['proceed'])

    def test_duplicate_names(self):
        rows=self.rows();rows[1]['name']=rows[0]['name']
        with self.assertRaises(AssertionError):summarize(rows)


if __name__=='__main__':unittest.main()

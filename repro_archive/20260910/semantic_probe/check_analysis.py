"""Regression checks for spatial accounting, label-free selection and stop gates."""
import copy
import unittest
import numpy as np
from analysis import SELECTORS, blocks, metrics, random_scores, selected_mask, summarize


class AnalysisChecks(unittest.TestCase):
    def test_exact_budget_and_row_major_ties(self):
        mask, ix = selected_mask(np.zeros(196))
        np.testing.assert_array_equal(ix, np.arange(40))
        np.testing.assert_array_equal(blocks(mask).sum(1), np.r_[np.full(40,256), np.zeros(156)])

    def test_noncontiguous_locations_are_not_transposed(self):
        scores = np.zeros(196); scores[17] = 1
        mask, ix = selected_mask(scores, k=1)
        self.assertTrue(mask[16:32,48:64].all())
        self.assertEqual(mask.sum(),256); self.assertEqual(ix.tolist(),[17])

    def test_invalid_scores_rejected(self):
        with self.assertRaises(AssertionError): selected_mask(np.full(196,np.nan))
        with self.assertRaises(AssertionError): selected_mask(np.zeros(195))

    def test_random_is_per_name_and_repeatable(self):
        np.testing.assert_array_equal(random_scores('a'),random_scores('a'))
        self.assertFalse(np.array_equal(random_scores('a'),random_scores('b')))

    def test_oracle_accounting_and_outside_identity(self):
        rng=np.random.default_rng(3)
        for _ in range(10):
            target=rng.random((224,224))>.8; pred=rng.random((224,224))>.7
            mask,_=selected_mask(rng.random(196))
            oracle=np.where(mask,target,pred)
            np.testing.assert_array_equal(oracle[~mask],pred[~mask])
            before,after=metrics(pred,target),metrics(oracle,target)
            self.assertGreaterEqual(after['iou'],before['iou'])
            self.assertEqual(before['fp']-after['fp'],int((mask&pred&~target).sum()))
            self.assertEqual(before['fn']-after['fn'],int((mask&~pred&target).sum()))

    def test_failed_capture_cannot_pass_on_iou_only(self):
        records=[]
        for i in range(8):
            baseline=dict(iou=.7,dice=.8,precision=.8,recall=.8,fp=20,fn=10,label_pixels=100+i)
            rows={s:dict(error_capture=.3,fp_captured=6,fn_captured=3,oracle_iou=.9,corrected=copy.deepcopy(baseline)) for s in SELECTORS}
            rows['disagreement']['corrected']['iou']+=.01
            records.append(dict(name=str(i),baseline=baseline,fine_probe=baseline,coarse_probe=baseline,full_correction=baseline,selectors=rows))
        self.assertFalse(summarize(records)['proceed_to_architecture'])
        for r in records:r['selectors']['disagreement']['error_capture']=.4
        self.assertTrue(summarize(records)['proceed_to_architecture'])
        records[0]['selectors']['disagreement']['corrected']['dice']-=.1
        self.assertFalse(summarize(records)['proceed_to_architecture'])


if __name__=='__main__': unittest.main()

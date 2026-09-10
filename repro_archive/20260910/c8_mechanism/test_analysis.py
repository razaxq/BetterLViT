"""Independent integer confusion-matrix fixtures for algebraic attribution."""
import unittest
from analyze_records import counts, decompose


class CountTests(unittest.TestCase):
    def test_integer_reconstruction(self):
        self.assertEqual(counts(dict(label_pixels=10, prediction_pixels=12,
            recall=.8, precision=8/12, iou=8/14, dice=16/22)),
            dict(tp=8, fp=4, fn=2, gt=10, pp=12))

    def test_only_false_positives_change(self):
        fp, fn = decompose(dict(gt=10,tp=8,fp=4),dict(gt=10,tp=8,fp=2))
        self.assertAlmostEqual(fp,8/12-8/14)
        self.assertEqual(fn,0)

    def test_only_false_negatives_change(self):
        fp, fn = decompose(dict(gt=10,tp=8,fp=4),dict(gt=10,tp=9,fp=4))
        self.assertEqual(fp,0)
        self.assertAlmostEqual(fn,1/14)

    def test_joint_changes_add_and_reverse(self):
        a,b=dict(gt=10,tp=8,fp=4),dict(gt=10,tp=9,fp=2)
        delta=decompose(a,b);reverse=decompose(b,a)
        self.assertAlmostEqual(sum(delta),9/12-8/14)
        for x,y in zip(delta,reverse): self.assertAlmostEqual(x,-y)

    def test_empty_ground_truth(self):
        fp, fn=decompose(dict(gt=0,tp=0,fp=2),dict(gt=0,tp=0,fp=0))
        self.assertEqual(fp,1)
        self.assertEqual(fn,0)


if __name__=='__main__': unittest.main()

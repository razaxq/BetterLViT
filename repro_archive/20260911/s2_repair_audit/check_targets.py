"""Boundary and geometry regression checks for the proposed target repair, CPU only."""
import json
import sys
import unittest
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0,'/root/autodl-tmp/BetterLViT-visual-aux-s2')
from race_semantics import make_zone_basis
from consistent_targets import full_resolution_region_targets


class RegionTests(unittest.TestCase):
    def setUp(self):self.basis=torch.from_numpy(make_zone_basis(224,224)).unsqueeze(0)
    def test_single_pixel_boundary(self):
        mask=torch.zeros(1,1,224,224);mask[0,0,74,10]=1
        ref=full_resolution_region_targets(mask,self.basis)
        self.assertEqual(ref['present'].tolist(),[[0,1,0,0,0,0]])
        low=F.adaptive_avg_pool2d(mask,28);old_basis=F.interpolate(self.basis,(28,28),mode='nearest')
        old=((low*old_basis).sum((2,3))>0).float()
        self.assertEqual(old.tolist(),[[1,0,0,0,0,0]])
    def test_augmented_geometry(self):
        mask=torch.zeros(1,1,224,224);mask[0,0,149,30]=1
        ref=full_resolution_region_targets(mask,self.basis)
        for k in (1,2,3):
            got=full_resolution_region_targets(torch.rot90(mask,k,(2,3)),torch.rot90(self.basis,k,(2,3)))
            self.assertTrue(torch.equal(got['present'],ref['present']))
            self.assertTrue(torch.equal(got['area'],ref['area']))
    def test_empty_invalid_and_full(self):
        basis=self.basis.clone();basis[:,0]=0
        got=full_resolution_region_targets(torch.zeros(1,1,224,224),basis)
        self.assertFalse(got['valid'][0,0]);self.assertEqual(got['present'].sum().item(),0)
        self.assertTrue(torch.isfinite(got['area']).all())
        full=full_resolution_region_targets(torch.ones(1,1,224,224),self.basis)
        self.assertTrue(torch.equal(full['area'],torch.ones(1,6)))
    def test_geometry_mismatch_rejected(self):
        with self.assertRaises(ValueError):full_resolution_region_targets(torch.zeros(1,1,28,28),self.basis)


if __name__=='__main__':
    torch.set_num_threads(2)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(RegionTests)
    result=unittest.TestResult();suite.run(result)
    print(json.dumps(dict(tests=result.testsRun,passed=result.wasSuccessful(),errors=[str(x) for x in result.errors],failures=[str(x) for x in result.failures])))
    raise SystemExit(0 if result.wasSuccessful() else 1)

"""Regression checks for legacy parity, augmentation RNG and epoch LR endpoints."""
import ast
import copy
import random
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import Config as config
import Load_Dataset as data
from training_recipe import SingleCosineSchedule, planned_rates
from utils import CosineAnnealingWarmRestarts


def np_state_equal(a,b):
    return a[0]==b[0] and np.array_equal(a[1],b[1]) and a[2:]==b[2:]


class RecipeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original=subprocess.check_output(['git','show','add4908a0d6f702b0a10c4581725b535543829b8:Load_Dataset.py'],text=True)
        node=next(n for n in ast.parse(original).body if isinstance(n,ast.ClassDef) and n.name=='RandomGenerator')
        ns=dict(vars(data))
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<frozen C4 transform>','exec'),ns)
        cls.legacy=ns['RandomGenerator']([224,224])
        grid=np.arange(224*224,dtype=np.uint32).reshape(224,224)
        cls.sample={'image':np.stack([grid%251,grid//224%251,grid%173],axis=-1).astype(np.uint8),
            'label':((grid%224>37)&(grid%224<93)&(grid//224>30)).astype(np.uint8),
            'race_zone_basis':np.stack([np.full((224,224),i/6,dtype=np.float32) for i in range(6)]),
            'input_ids':torch.arange(24),'attention_mask':torch.ones(24,dtype=torch.long),
            'race_slot_targets':torch.zeros(6)}

    def run_transform(self,transform,policy,seed,draws=None):
        random.seed(seed); np.random.seed(seed)
        with patch.object(config,'augmentation_policy',policy):
            if draws is None:
                output=transform(copy.deepcopy(self.sample))
            else:
                with patch.object(data.random,'random',side_effect=draws):
                    output=transform(copy.deepcopy(self.sample))
        return output, random.getstate(), np.random.get_state()

    def assert_outputs(self,a,b):
        self.assertEqual(a.keys(),b.keys())
        for key in a:self.assertTrue(torch.equal(a[key],b[key]),key)

    def test_legacy_transform_matches_frozen_c4(self):
        for seed in range(40):
            a,pa,na=self.run_transform(self.legacy,'legacy',seed)
            b,pb,nb=self.run_transform(data.RandomGenerator([224,224]),'legacy',seed)
            self.assert_outputs(a,b);self.assertEqual(pa,pb);self.assertTrue(np_state_equal(na,nb))

    def test_removed_branch_identity_and_rng(self):
        a,pa,na=self.run_transform(data.RandomGenerator([224,224]),'legacy',13,[.75])
        b,pb,nb=self.run_transform(data.RandomGenerator([224,224]),'chest_orientation',13,[.75])
        expected=data.ValGenerator([224,224])(copy.deepcopy(self.sample))
        self.assert_outputs(b,expected)
        self.assertFalse(torch.equal(a['image'],b['image']))
        self.assertEqual(pa,pb);self.assertTrue(np_state_equal(na,nb))

    def test_other_branches_and_future_rng_unchanged(self):
        for draws in ([.25,.75],[.25,.25]):
            a,pa,na=self.run_transform(data.RandomGenerator([224,224]),'legacy',17,draws)
            b,pb,nb=self.run_transform(data.RandomGenerator([224,224]),'chest_orientation',17,draws)
            self.assert_outputs(a,b);self.assertEqual(pa,pb);self.assertTrue(np_state_equal(na,nb))
        # Unforced branch sequences retain both PRNG streams across real draws.
        for seed in range(20):
            _,pa,na=self.run_transform(data.RandomGenerator([224,224]),'legacy',seed)
            _,pb,nb=self.run_transform(data.RandomGenerator([224,224]),'chest_orientation',seed)
            self.assertEqual(pa,pb);self.assertTrue(np_state_equal(na,nb))

    def test_schedule_endpoints_and_resume(self):
        opt=torch.optim.Adam([torch.nn.Parameter(torch.ones(1))],lr=3e-4)
        schedule=SingleCosineSchedule(opt,80)
        rates=[]
        for epoch in range(80):
            rates.append(opt.param_groups[0]['lr'])
            if epoch==36:saved=copy.deepcopy(schedule.state_dict())
            schedule.step()
        self.assertEqual(rates,planned_rates('single_cosine'))
        self.assertEqual(rates[0],3e-4);self.assertEqual(rates[-1],1e-6)
        self.assertTrue(all(a>=b for a,b in zip(rates,rates[1:])))
        other=torch.optim.Adam([torch.nn.Parameter(torch.ones(1))],lr=3e-4)
        resumed=SingleCosineSchedule(other,80);resumed.load_state_dict(saved)
        for epoch in range(36,80):
            self.assertEqual(other.param_groups[0]['lr'],rates[epoch]);resumed.step()
        self.assertEqual(other.param_groups[0]['lr'],1e-6)

    def test_legacy_scheduler_sequence_unchanged(self):
        opt=torch.optim.Adam([torch.nn.Parameter(torch.ones(1))],lr=3e-4)
        schedule=CosineAnnealingWarmRestarts(opt,T_0=10,T_mult=1,eta_min=1e-4)
        for expected in planned_rates('warm_restarts'):
            self.assertAlmostEqual(opt.param_groups[0]['lr'],expected,places=15);schedule.step()


if __name__=='__main__':
    unittest.main()

"""Causal-path and target-semantics tests with independent fixtures."""
import copy
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import torch
from nets.visual_aux import VisualAuxiliaryHeads
from visual_aux_objective import VisualAuxiliaryObjective
from training_recipe import planned_rates
from visual_aux_protocol import validate_visual_aux_manifest


class VisualAuxTests(unittest.TestCase):
    def fixture(self,mode):
        torch.manual_seed(17)
        head=VisualAuxiliaryHeads(mode,channels=(8,16),hidden_channels=4)
        features=(torch.randn(2,8,8,8,requires_grad=True),torch.randn(2,16,4,4,requires_grad=True))
        basis=torch.zeros(2,6,8,8);basis[:,0,:,:4]=1;basis[:,1,:,4:]=1
        skips,out=head(features,None,None,basis)
        out['final']=torch.full((2,1,8,8),.4,requires_grad=True)
        gt=torch.zeros(2,8,8);gt[0,1,1]=1;gt[1]=1
        return head,features,skips,out,gt

    def test_identity_and_no_text_dependency(self):
        h,f,s,o,g=self.fixture('regional')
        self.assertTrue(all(a is b for a,b in zip(f,s)))
        self.assertNotIn('slot_logits',o)
        self.assertTrue(all(not p.requires_grad for p in h.slot_head.parameters()))

    def test_pixel_only_has_no_presence_gradient(self):
        h,f,s,o,g=self.fixture('pixel')
        losses=VisualAuxiliaryObjective('pixel').components(o,g)
        self.assertEqual(set(losses),{'main','pixel'})
        sum(losses.values()).backward()
        self.assertGreater(h.routes[0].evidence[0].weight.grad.abs().sum().item(),0)
        self.assertIsNone(h.routes[0].presence.weight.grad)

    def test_regional_head_receives_gradients(self):
        h,f,s,o,g=self.fixture('regional')
        VisualAuxiliaryObjective('regional')(o,g).backward()
        self.assertGreater(h.routes[0].presence.weight.grad.abs().sum().item(),0)
        self.assertTrue(all(torch.isfinite(x.grad).all() for x in f))
        self.assertTrue(all(p.grad is None for p in h.slot_head.parameters()))

    def test_effective_weights_do_not_renormalize(self):
        h,f,s,o,g=self.fixture('regional')
        for route in o['pe_routes']:
            route['extent_logits']=torch.zeros_like(route['extent_logits'])
            route['presence_logits']=torch.zeros_like(route['presence_logits'])
            route['occupancy']=torch.full_like(route['occupancy'],.5)
        c=VisualAuxiliaryObjective('regional').components(o,g)
        self.assertAlmostEqual(c['pixel'].item(),.02*math.log(2),places=7)
        self.assertAlmostEqual(c['presence'].item(),.01*math.log(2),places=7)
        p=VisualAuxiliaryObjective('pixel').components(o,g)
        self.assertEqual(c['pixel'].item(),p['pixel'].item())

    def test_invalid_regions_empty_mask_are_finite(self):
        h,f,s,o,g=self.fixture('regional')
        for route in o['pe_routes']: route['basis'].zero_()
        c=VisualAuxiliaryObjective('regional').components(o,torch.zeros_like(g))
        self.assertEqual(c['presence'].item(),0)
        self.assertEqual(c['occupancy'].item(),0)
        self.assertTrue(torch.isfinite(sum(c.values())))

    def test_rng_isolation_and_common_head_initialization(self):
        torch.manual_seed(33);before=torch.get_rng_state().clone()
        with torch.random.fork_rng(devices=[]):
            a=VisualAuxiliaryHeads('pixel',channels=(8,16),hidden_channels=4)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        with torch.random.fork_rng(devices=[]):
            b=VisualAuxiliaryHeads('regional',channels=(8,16),hidden_channels=4)
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))

    def test_presence_is_any_foreground_not_extent(self):
        h,f,s,o,g=self.fixture('regional')
        # A single pixel is present but occupies only 1/32 of this region.
        area=(g.unsqueeze(1)*o['pe_routes'][0]['basis']).sum((2,3))/o['pe_routes'][0]['basis'].sum((2,3)).clamp_min(1)
        self.assertEqual(area[0,0].item(),1/32)
        self.assertEqual((area>0)[0,0].item(),True)
        self.assertEqual((area>0)[0,1].item(),False)


if __name__=='__main__':unittest.main()

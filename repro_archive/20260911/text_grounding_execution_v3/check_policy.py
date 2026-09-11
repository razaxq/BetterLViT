"""Behavioral checks for semantic controls and independent metric conventions."""
import unittest
from text_policy import TextPolicy,parse,render,group
from analysis import metrics,interval,summarize
from routing import separate_forward,eppa_text_override
from data_identity import build_bindings


class Checks(unittest.TestCase):
    def test_loader_and_workbook_names_are_distinct(self):
        image='sub-S10644_ses-E19682_run-1_bp-chest_vp-ap_cr.png'
        mask='mask_'+image
        text='Unilateral pulmonary infection, one infected area, lower right lung.'
        bindings=build_bindings([mask,'covid_1.png'],{mask:text,'covid_1.png':'Pulmonary infection.'})
        self.assertEqual(bindings[image],dict(mask_name=mask,text=text))
        self.assertEqual(bindings['covid_1.png']['mask_name'],'covid_1.png')
        self.assertEqual(TextPolicy([text]).variants(bindings[image]['mask_name'],bindings[image]['text']),
                         TextPolicy([text]).variants(mask,text))

    def test_ambiguous_or_missing_identity_fails(self):
        with self.assertRaises(ValueError):build_bindings(['a.png','mask_a.png'],{'a.png':'one','mask_a.png':'two'})
        with self.assertRaises(KeyError):build_bindings(['mask_a.png'],{'a.png':'not the workbook key'})

    def test_binding_not_bag_of_words(self):
        a='Bilateral pulmonary infection, two infected areas, upper left lung and lower right lung.'
        b='Bilateral pulmonary infection, two infected areas, lower left lung and upper right lung.'
        self.assertEqual(sorted(a.lower().split()),sorted(b.lower().split()))
        self.assertNotEqual(parse(a),parse(b))
        self.assertEqual(parse(TextPolicy([a]).variants('a',a)['relation_swap']),parse(b))

    def test_symmetric_is_identity(self):
        t='Bilateral pulmonary infection, two infected areas, all left lung and all right lung.'
        self.assertEqual(TextPolicy([t]).variants('a',t)['relation_swap'],t)
        self.assertEqual(group(parse(t)),'symmetric_bilateral')

    def test_unilateral_does_not_become_bilateral(self):
        t='Unilateral pulmonary infection, one infected area, lower right lung.'
        v=TextPolicy([t]).variants('a',t)
        self.assertEqual(parse(v['relation_swap'])[2],(('left',('lower',)),))
        self.assertEqual(parse(v['relation_swap'])[:2],parse(t)[:2])

    def test_unknown_and_contradictory_abstain(self):
        for t in ['No upper left infection.', 'Unilateral pulmonary infection, one infected area, lower left lung and lower right lung.',
                  'Bilateral pulmonary infection, two infected areas, lowerl left lung and upper right lung.']:
            self.assertIsNone(parse(t))
            v=TextPolicy([]).variants('a',t)
            self.assertEqual(v['relation_swap'],t)
            self.assertEqual(v['canonical'],t)

    def test_semantic_match_uses_train_only(self):
        a='Unilateral pulmonary infection, one infected area, all right lung.'
        b='Unilateral pulmonary infection, one infected area, upper middle lower right lung.'
        p=TextPolicy([a,b])
        self.assertEqual(p.variants('n',a)['semantic_match'],b)
        self.assertEqual(parse(render(parse(a))),parse(a))
        self.assertEqual(p.variants('n',a),p.variants('n',a))

    def test_threshold_and_counts(self):
        r=metrics([.5,.6,.4,.8],[1,1,0,0])
        self.assertEqual((r['tp'],r['fp'],r['fn']),(1,1,1))
        self.assertEqual(r['iou'],1/3)
        self.assertEqual(r['dice'],.5)
        self.assertEqual(interval([0.,0.]),[0.,0.])

    def test_binding_contrast_uses_canonical_control(self):
        base=metrics([.8,.2],[1,0])
        canonical=dict(metrics([.8,.8],[1,0]),tokens_changed=True,probability_mae=.3,changed_pixels=1)
        swapped=dict(metrics([.2,.8],[1,0]),tokens_changed=True,probability_mae=.6,changed_pixels=2)
        conditions={v+'/'+s:(swapped if v=='relation_swap' else canonical)
                    for v in ('canonical','semantic_match','relation_swap','drop_location','generic')
                    for s in ('main','eppa','both')}
        r=dict(name='fixture',group='unilateral',train_template_frequency=1,baseline=base,conditions=conditions)
        summary=summarize([r])
        self.assertEqual(summary['binding_controlled_contrast']['main']['swap_minus_semantic_canonical']['iou']['mean'],-.5)
        self.assertFalse(summary['automatic_architecture_pass'])

    def test_early_launch_never_connects(self):
        from unittest.mock import patch
        import control
        with patch('control.time.time',return_value=control.MANIFEST['minimum_launch_unix']-1),patch('control.remote') as remote:
            with self.assertRaises(AssertionError):control.launch()
            remote.assert_not_called()

    def test_branch_separation_and_restoration(self):
        class Block:
            def forward(self, x, text=None):return x+text
        class Up:
            def __init__(self):self.eppa=Block()
        class Model:
            def __init__(self):
                for name in ('up4','up3','up2','up1'):setattr(self,name,Up())
        def forward(model,x,text,text_mask=None):
            for name in ('up4','up3','up2','up1'):x=getattr(model,name).eppa.forward(x,text=text)
            return x+10*text
        model=Model()
        self.assertEqual(separate_forward(forward,model,0,2,None,3),32)
        self.assertEqual(forward(model,0,2),28)
        try:
            with eppa_text_override(model,99):raise ValueError('fixture')
        except ValueError:pass
        self.assertEqual(forward(model,0,2),28)
        self.assertTrue(all('forward' not in getattr(model,n).eppa.__dict__ for n in ('up4','up3','up2','up1')))


if __name__=='__main__':unittest.main()

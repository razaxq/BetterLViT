"""V2 semantics and gradient directions, including conflicting reports/masks."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nets.race_pe import RACEPEV2
from race_pe_objective import RACEPEObjective
from race_semantics import parse_report_mentions,make_zone_basis


def main():
    assert parse_report_mentions('one left upper opacity')[:6].tolist()==[1,0,0,0,0,0]
    assert parse_report_mentions('lower left lung and all right lung')[:6].tolist()==[0,0,1,1,1,1]
    assert parse_report_mentions('opacities')[:6].eq(0).all()
    assert parse_report_mentions('possible left upper opacity')[:6].eq(-1).all()
    torch.manual_seed(1219)
    m=RACEPEV2(channels=(8,),hidden_channels=8)
    x=(torch.randn(2,8,24,24),); text=torch.randn(2,12,768)
    basis=torch.from_numpy(make_zone_basis(24,24)).repeat(2,1,1,1)
    routed,a=m(x,text,None,basis)
    assert torch.equal(x[0],routed[0])
    a.update(final=torch.full((2,1,24,24),.3,requires_grad=True),
        race_slot_targets=torch.stack([parse_report_mentions('left upper opacity')]*2),race_zone_basis=basis)
    a['slot_logits'].retain_grad();a['pe_routes'][0]['presence_logits'].retain_grad()
    criterion=RACEPEObjective(drop_report_consistency=True)
    loss=criterion(a,torch.zeros(2,24,24));loss.backward()
    # Non-mentioned is a valid negative ONLY for the mention extraction task.
    assert (a['slot_logits'].grad[:,1:6]>0).all()
    assert (a['slot_logits'].grad[:,0]<0).all()
    # Empty mask must push presence down even if report mentions that region.
    assert (a['pe_routes'][0]['presence_logits'].grad>0).all()
    print(json.dumps({'status':'ok','identity_error':0,'negative_mention_gradient':'down',
        'positive_mention_gradient':'up','conflicting_empty_region_gradient':'down'}))


if __name__=='__main__':main()

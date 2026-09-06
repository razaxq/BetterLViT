"""Behavioral checks for extent supervision, unknown semantics and routing."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from nets.race_pe import RACEPE
from race_pe_objective import RACEPEObjective
from race_semantics import make_zone_basis, parse_report_slots_pe
from checkpoint_selection import is_improvement


def main():
    torch.manual_seed(1219)
    torch.use_deterministic_algorithms(True)
    model = RACEPE(channels=(8, 16), hidden_channels=8)
    skips = (torch.randn(2, 8, 24, 24), torch.randn(2, 16, 12, 12))
    text = torch.randn(2, 12, 768)
    basis = torch.from_numpy(make_zone_basis(24, 24)).repeat(2, 1, 1, 1)
    out, aux = model(skips, text, None, basis)
    again, _ = model(skips, text, None, basis)
    assert all(torch.equal(x, y) and torch.equal(y, z) for x, y, z in zip(skips, out, again))
    labels = torch.stack([parse_report_slots_pe('one left upper opacity'),
                          parse_report_slots_pe('opacities')])
    assert (labels[1] == -1).all()
    assert parse_report_slots_pe('no left upper opacity').eq(-1).all()
    pair = parse_report_slots_pe('left upper and right lower opacities')[:6]
    assert pair.tolist() == [1, -1, -1, -1, -1, 1]
    assert parse_report_slots_pe('Bilateral infection, lower right lung')[:6].tolist() == [-1, -1, -1, -1, -1, 1]
    mask = torch.zeros(2, 24, 24)
    mask[0, 2, 2] = 1  # A single-pixel lesion must survive scale reduction.
    aux.update(final=torch.full((2, 1, 24, 24), 0.4, requires_grad=True),
               race_slot_targets=labels, race_zone_basis=basis)
    for row in aux['pe_routes']:
        row['extent_logits'].retain_grad()
        row['presence_logits'].retain_grad()
    aux['slot_logits'].retain_grad()
    objective = RACEPEObjective()
    loss = objective(aux, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert aux['slot_logits'].grad[1].eq(0).all(), 'Unknown text must not act as negative'
    for row in aux['pe_routes']:
        assert torch.isfinite(row['extent_logits'].grad).all()
        assert row['extent_logits'].grad[1].mean() > 0, 'Empty mask must suppress extent'
        assert row['presence_logits'].grad[0, 0] < 0, 'Tiny lesion must train presence positive'
        assert row['presence_logits'].grad[1].mean() > 0
    # Distinct objectives: low occupancy does not prohibit high existence.
    history = [{'epoch': 6, 'val_dice': .8, 'val_iou': .7},
               {'epoch': 7, 'val_dice': .79, 'val_iou': .71}]
    assert is_improvement(history, 'iou') and not is_improvement(history, 'dice')
    from compare_race_pe import passes
    good = {'overall': {'iou': {'mean_delta': .004}, 'dice': {'mean_delta': .002},
            'precision': {'mean_delta': 0}, 'brier_lower_is_better': {'mean_delta': -.001}},
            'lesion_size_quartiles': [{'dice': {'mean_delta': 0}, 'recall': {'mean_delta': 0}}]}
    assert passes(good)
    good['overall']['iou']['mean_delta'] = .0029
    assert not passes(good)
    with torch.no_grad():
        for route in model.routes:
            route.strength_logit.fill_(0.2)
    routed, _ = model(skips, text, None, torch.zeros_like(basis))
    assert all(torch.equal(x, y) for x, y in zip(skips, routed)), 'Invalid zones must not route'
    model.route_enabled = False
    routed, _ = model(skips, text, None, basis)
    assert all(torch.equal(x, y) for x, y in zip(skips, routed))
    print(json.dumps({'status': 'ok', 'identity_error': 0, 'repeat_error': 0,
                      'loss': loss.item(), 'checks': ['unknown', 'negation', 'tiny_lesion',
                      'empty_mask', 'invalid_zones', 'aux_only', 'iou_selection']}))


if __name__ == '__main__':
    main()

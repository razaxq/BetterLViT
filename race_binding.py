"""Explicit report mention binding for the matched P8 repair control.

Outputs describe explicit report mentions, never disease absence. Uncovered
grammar is unknown. Count labels and spatial bases are deliberately untouched.
"""
import re

ZONES=('left_upper','left_middle','left_lower','right_upper','right_middle','right_lower')

def explicit_location_mentions(report):
    normalized=' '.join(str(report).lower().replace('-',' ').split())
    tail=normalized.rsplit(',',1)[-1].strip().rstrip('.')
    clauses=re.split(r'\s+and\s+',tail)
    matches=[re.fullmatch(r'((?:(?:all|upper|middle|lower)\s+)+)(left|right)\s+lung',c.strip()) for c in clauses]
    if not matches or not all(matches):
        return dict(known=False,mentioned_zones=None,reason='uncovered_or_ambiguous_grammar')
    mentions=[0]*6
    for match in matches:
        levels=match.group(1).split()
        offset=0 if match.group(2)=='left' else 3
        for i,level in enumerate(('upper','middle','lower')):
            if 'all' in levels or level in levels:mentions[offset+i]=1
    return dict(known=True,mentioned_zones=mentions,reason='exact_side_bound_clauses')

def check_behavior():
    # Independent linguistic examples, including missing, negated and ambiguous text.
    cases={
        'all left lung and middle lower right lung.':[1,1,1,0,1,1],
        'lower left lung and upper right lung.':[0,0,1,1,0,0],
        'upper middle lower left lung and all right lung.':[1,1,1,1,1,1],
        'Bilateral pulmonary infection, two infected areas, lower left lung.':[0,0,1,0,0,0],
    }
    for report,expected in cases.items():
        assert explicit_location_mentions(report)['mentioned_zones']==expected
    for report in ('pulmonary infection.','no left lung infection.','left lung may be affected.',''):
        result=explicit_location_mentions(report)
        assert not result['known'] and result['mentioned_zones'] is None
    a='lower left lung and upper right lung.'
    b='upper right lung and lower left lung.'
    assert explicit_location_mentions(a)==explicit_location_mentions(b)
    swapped=a.replace('left','TEMP').replace('right','left').replace('TEMP','right')
    old=explicit_location_mentions(a)['mentioned_zones']
    assert explicit_location_mentions(swapped)['mentioned_zones']==old[3:]+old[:3]
    print('10 report-binding semantic and metamorphic checks passed.')

if __name__=='__main__':check_behavior()


def parse_report_slots_binding(text):
    """Repair only covered six-slot bindings; retain legacy count and fallback.

    Zero still means unmentioned in the existing slot-head BCE, not clinical
    absence. Unknown masking is deliberately a separate, untested change.
    """
    import torch
    from race_semantics import parse_report_slots
    slots = parse_report_slots(text)
    result = explicit_location_mentions(text)
    if result['known']:
        slots[:6] = torch.tensor(result['mentioned_zones'], dtype=slots.dtype)
    return slots

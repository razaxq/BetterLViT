"""Semantic regression and frozen-profile checks, without inference or Test."""
import json,runpy,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from race_binding import check_behavior,parse_report_slots_binding,explicit_location_mentions
from race_semantics import parse_report_slots
from paper_experiments import get_paper_experiment
from p8_restart_protocol import validate_p8_manifest

check_behavior()
reports=['all left lung and middle lower right lung.','lower left lung and upper right lung.',
    'pulmonary infection.','no left lung infection.','left lung may be affected.','']
for prefix in ('one infected area, ','two infected areas, ','three infected areas, ','four infected areas, '):
    for report in reports:
        text=prefix+report;old=parse_report_slots(text);new=parse_report_slots_binding(text)
        assert torch.equal(old[6:],new[6:]),'Count repair was not authorized in this contrast'
        if not explicit_location_mentions(text)['known']:assert torch.equal(old,new)
        assert ((new==0)|(new==1)).all(),'No hidden unknown-mask change'
assert parse_report_slots_binding(reports[0])[:6].tolist()==[1,1,1,0,1,1]
original=get_paper_experiment('p8_r2_original');binding=get_paper_experiment('p8_r2_binding')
allowed={'name','paper_id','description','architecture_version','race_binding_repair'}
assert all(original[k]==binding[k] for k in original.keys()-allowed)
manifest=json.loads((Path(__file__).resolve().parents[1]/'experiment_manifests/active_p8_restart.json').read_text())
validate_p8_manifest(manifest)
print(json.dumps(dict(status='pass',count_and_fallback_cases=24,profile=manifest['profile'],test_split_accessed=False)))

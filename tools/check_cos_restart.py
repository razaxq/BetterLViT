"""Test actual warm restart rates, immutable profiles, CLI and metric equivalence."""
import argparse,ast,copy,json,subprocess,sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from paper_experiments import PAPER_EXPERIMENTS
from cos_restart_protocol import PROFILES,validate_manifest
from training_recipe import planned_rates,rates_equal

def main():
    old={};exec(subprocess.check_output(['git','show','014f42cf084dab01ab4e771a511efcec84276261:paper_experiments.py'],cwd=ROOT),old)
    assert all(PAPER_EXPERIMENTS[k]==v for k,v in old['PAPER_EXPERIMENTS'].items())
    ignored={'paper_id','description','architecture_version','lr_schedule'}
    for new,parent in [('cr0_plam','j0_plam_r2'),('cr3_fsdr_race_binding','p8_r2_binding')]:
        assert {k:v for k,v in PAPER_EXPERIMENTS[new].items() if k not in ignored}=={k:v for k,v in PAPER_EXPERIMENTS[parent].items() if k not in ignored}
    m=validate_manifest(json.loads((ROOT/'experiment_manifests/active_cos_restart.json').read_text()))
    for key,value in [('epochs',150),('threshold',.4),('race_aux_weight',.01),('seed',2027),('lr_schedule','single_cosine')]:
        invalid=copy.deepcopy(m);invalid[key]=value
        try:validate_manifest(invalid)
        except (AssertionError,KeyError):pass
        else:raise AssertionError('Invalid manifest accepted: '+key)
    import torch
    from utils import CosineAnnealingWarmRestarts
    optimizer=torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))],lr=3e-4)
    scheduler=CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=1,eta_min=1e-4)
    rates=[]
    for _ in range(80):
        rates.append(optimizer.param_groups[0]['lr']);optimizer.step();scheduler.step()
    assert rates_equal(rates,planned_rates('warm_restarts'))
    assert all(abs(rates[i]-3e-4)<1e-15 for i in range(0,80,10))
    t=ast.parse((ROOT/'tools/export_cos_restart_metrics.py').read_text())
    nodes=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='parse_args']
    scope=dict(argparse=argparse,Path=Path,ALLOWED_EXPERIMENTS=tuple(PROFILES))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual exporter CLI>','exec'),scope)
    for profile in PROFILES:
        for split in ('validation','test'):
            with patch.object(sys,'argv',['export','--experiment',profile,'--checkpoint','best','--output','result','--split',split,'--expected-best-epoch','75']):
                a=scope['parse_args']();assert a.experiment==profile and a.split==split
    original=ast.parse((ROOT/'tools/export_stage1_metrics.py').read_text())
    for name in ('mask_boundary','boundary_f1','image_frequency_scores','validation_loader'):
        a=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(a)==ast.dump(b),name
    new=(ROOT/'tools/export_cos_restart_metrics.py').read_text();oldtext=(ROOT/'tools/export_stage1_metrics.py').read_text()
    assert new[new.index('    records ='):new.index('    result =')]==oldtext[oldtext.index('    records ='):oldtext.index('    result =')]
    print(json.dumps(dict(status='pass',historical_profiles_unchanged=True,only_lr_strategy_changed=True,actual_scheduler_all_80_rates_verified=True,manifest_rejection_checks=True,all_four_cli_modes=True,metric_loop_and_helpers_identical=True,test_split_accessed=False)))

if __name__=='__main__':main()

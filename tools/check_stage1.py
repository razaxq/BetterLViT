"""Recipe scope, historical profile invariance and actual exporter CLI regression."""
import argparse,ast,copy,json,subprocess,sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from paper_experiments import PAPER_EXPERIMENTS
from stage1_protocol import PROFILES,validate_manifest

def main():
    old={};exec(subprocess.check_output(['git','show','ec9f3590d535ee29c02484e3c4babc6fb0c099d0:paper_experiments.py'],cwd=ROOT),old)
    assert all(PAPER_EXPERIMENTS[k]==v for k,v in old['PAPER_EXPERIMENTS'].items())
    m=validate_manifest(json.loads((ROOT/'experiment_manifests/active_stage1.json').read_text()))
    for key,value in [('epochs',79),('threshold',.4),('race_aux_weight',.01),('seed',17)]:
        invalid=copy.deepcopy(m);invalid[key]=value
        try:validate_manifest(invalid)
        except (AssertionError,KeyError):pass
        else:raise AssertionError('Invalid manifest accepted: '+key)
    t=ast.parse((ROOT/'tools/export_stage1_metrics.py').read_text())
    nodes=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='parse_args']
    scope=dict(argparse=argparse,Path=Path,ALLOWED_EXPERIMENTS=tuple(PROFILES))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual stage1 CLI>','exec'),scope)
    for profile in PROFILES:
        for split in ('validation','test'):
            with patch.object(sys,'argv',['export','--experiment',profile,'--checkpoint','best','--output','result','--split',split,'--expected-best-epoch','75']):
                a=scope['parse_args']();assert a.experiment==profile and a.split==split
    original=ast.parse((ROOT/'tools/export_validation_metrics.py').read_text())
    for name in ('mask_boundary','boundary_f1','image_frequency_scores','validation_loader'):
        a=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(a)==ast.dump(b),name
    # Preserve the complete metric loop and aggregation, not just helper names.
    new=(ROOT/'tools/export_stage1_metrics.py').read_text();oldtext=(ROOT/'tools/export_validation_metrics.py').read_text()
    a=new[new.index('    records ='):new.index('    result =')]
    b=oldtext[oldtext.index('    records ='):oldtext.index('    result =')]
    assert a==b
    print(json.dumps(dict(status='pass',historical_profiles_unchanged=True,manifest_rejection_checks=True,all_six_cli_modes=True,metric_loop_and_helpers_identical=True)))

if __name__=='__main__':main()

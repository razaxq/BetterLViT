"""Regression: old module parity, strict identity bypass, live auxiliary gradients, CLI."""
import argparse,ast,json,subprocess,sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from nets.race_fuse import RACEFuse
from race_semantics import make_zone_basis
from utils import RACEObjective
from p8_aux_protocol import validate_manifest
validate_manifest(json.loads((ROOT/'experiment_manifests/active_aux_only.json').read_text()))
ns={};exec(subprocess.check_output(['git','show','c6e52a9b13576cab9e2451c550d03ad274bc09e2:nets/race_fuse.py'],cwd=ROOT),ns)
torch.manual_seed(1219);old=ns['RACEFuse'](channels=(8,16,24,32))
torch.manual_seed(1219);new=RACEFuse(channels=(8,16,24,32))
assert old.state_dict().keys()==new.state_dict().keys()
assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
skips=[torch.randn(2,c,24//2**i,24//2**i,requires_grad=True) for i,c in enumerate((8,16,24,32))]
text=torch.randn(2,12,768);basis=torch.from_numpy(make_zone_basis(24,24)).unsqueeze(0).repeat(2,1,1,1)
a,auxa=old(skips,text,None,basis);b,auxb=new(skips,text,None,basis)
assert all(torch.equal(x,y) for x,y in zip(a,b))
assert torch.equal(auxa['slot_logits'],auxb['slot_logits'])
assert all(torch.equal(x,y) for x,y in zip(auxa['visual_zone_probabilities'],auxb['visual_zone_probabilities']))
new.route_enabled=False
with torch.no_grad():
    for route in new.routes:route.strength_logit.fill_(2.)
b,aux=new(skips,text,None,basis)
assert all(x is y for x,y in zip(skips,b)),'Disabled routing must be strict input identity'
target=(torch.rand(2,24,24)>.7).float();slots=torch.zeros(2,9);slots[:,[0,2,4,7]]=1
aux.update(final=torch.sigmoid(torch.randn(2,1,24,24)),race_slot_targets=slots,race_zone_basis=basis)
loss=RACEObjective(.05)(aux,target);loss.backward()
assert all(x.grad is not None and x.grad.abs().max()>0 for x in skips)
assert new.slot_head[0].weight.grad.abs().max()>0
for route in new.routes:
    assert route.evidence[-1].weight.grad.abs().max()>0
    assert route.strength_logit.grad is None
    assert all(p.grad is None for p in route.residual.parameters())
tree=ast.parse((ROOT/'tools/export_validation_metrics.py').read_text())
nodes=[n for n in tree.body if (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ALLOWED_EXPERIMENTS' for t in n.targets)) or (isinstance(n,ast.FunctionDef) and n.name=='parse_args')]
space=dict(argparse=argparse,Path=Path);exec(compile(ast.Module(body=nodes,type_ignores=[]),'<real exporter CLI>','exec'),space)
for split in ('validation','test'):
    with patch.object(sys,'argv',['export','--experiment','p8_r2_binding_aux','--checkpoint','best','--output','result','--split',split,'--expected-best-epoch','75']):
        args=space['parse_args']();assert args.experiment=='p8_r2_binding_aux' and args.split==split
print(json.dumps(dict(status='pass',historical_init_and_default_output_exact=True,route_disabled_identity=True,
    auxiliary_encoder_and_head_gradients=True,residual_and_strength_no_grad=True,both_evaluation_cli_modes=True)))

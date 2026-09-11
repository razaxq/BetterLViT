"""CPU-only strict validation of all 20 completed fold heads before archiving."""
import json
from pathlib import Path
from control import remote
from analysis import write_json
HERE=Path(__file__).resolve().parent
d=json.loads((HERE/'deployment.json').read_text())
proof=json.loads((HERE/'results/independent_verification.json').read_text());assert proof['verified']
result=remote('ROOT='+repr(d['remote_directory'])+'\nSHA='+repr(d['source_git_commit'])+'\n'+'''
import hashlib,json,sys
from pathlib import Path
root=Path(ROOT);sys.path.insert(0,str(root))
import numpy as np,torch
from refiner import LocalRefiner,VARIANTS
runtime=json.loads((root/'results/runtime.json').read_text());assert runtime['phase']=='complete'
provenance=json.loads((root/'results/provenance.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
verified={}
for fold in range(5):
    name=f'fold_{fold}_heads.pt';path=root/'results'/name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==runtime['artifacts'][name]['sha256']
    value=torch.load(path,map_location='cpu',weights_only=True)
    assert value['source_git_commit']==SHA and value['fold']==fold and value['steps']==2048 and value['manifest']==manifest
    assert set(value['heads'])==set(VARIANTS) and set(value['fit_indices']).isdisjoint(value['held_indices'])
    for variant,state in value['heads'].items():
        model=LocalRefiner();model.load_state_dict(state,strict=True)
        assert all(torch.isfinite(p).all() for p in model.state_dict().values())
        h=hashlib.sha256()
        for key,tensor in sorted(model.state_dict().items()):h.update(key.encode());h.update(tensor.detach().numpy().tobytes())
        assert h.hexdigest()==provenance['final_state_sha256'][str(fold)][variant]
        verified[str(fold)+'/'+variant]=h.hexdigest()
resume=torch.load(root/'results/resume.pt',map_location='cpu',weights_only=True)
assert resume['source_git_commit']==SHA and resume['fold']==4 and resume['step']==2048 and resume['manifest']==manifest
for variant,state in resume['heads'].items():
    model=LocalRefiner();model.load_state_dict(state,strict=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    optimizer.load_state_dict(resume['optimizers'][variant])
    assert all(torch.isfinite(t).all() for s in optimizer.state.values() for t in s.values() if torch.is_tensor(t))
print(json.dumps(dict(verified=True,source_git_commit=SHA,folds=5,heads=20,steps_per_head=2048,
    strict_load_cpu=True,weights_only=True,all_finite=True,state_sha256=verified,resume_optimizer_load_verified=True)))
''')
write_json(HERE/'checkpoint_verified.json',result);print(json.dumps(result))

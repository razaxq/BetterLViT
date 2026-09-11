"""Read-only CPU load verification of the saved D heads; no run-status query."""
import json
from pathlib import Path
from control import remote,HERE
from analysis import write_json,digest

deployment=json.loads((HERE/'deployment.json').read_text())
runtime=json.loads((HERE/'results/runtime.json').read_text())
result=remote('ROOT='+repr(deployment['remote_directory'])+'\nSHA='+repr(runtime['source_git_commit'])+'\nEXPECTED='+repr(runtime['artifacts']['heads.pt']['sha256'])+'\n'+'''
import hashlib,json,sys
from pathlib import Path
import torch
root=Path(ROOT);sys.path.insert(0,str(root))
from heads import ResidualHead,VARIANTS
torch.set_num_threads(4)
path=root/'results/heads.pt'
assert hashlib.sha256(path.read_bytes()).hexdigest()==EXPECTED
checkpoint=torch.load(path,map_location='cpu',weights_only=True)
assert checkpoint['source_git_commit']==SHA and checkpoint['steps']==512
assert checkpoint['manifest']==json.loads((root/'manifest.json').read_text())
expected={t+'/'+v for t in checkpoint['manifest']['treatments'] for v in VARIANTS}
assert set(checkpoint['heads'])==expected
hashes={}
for key,state in checkpoint['heads'].items():
    model=ResidualHead();model.load_state_dict(state,strict=True)
    assert all(torch.isfinite(value).all() for value in state.values())
    assert sum(p.numel() for p in model.parameters())==75808
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode());h.update(value.detach().contiguous().numpy().tobytes())
    hashes[key]=h.hexdigest()
print(json.dumps(dict(verified=True,source_git_commit=SHA,checkpoint_sha256=EXPECTED,heads=18,steps_per_head=512,
    strict_state_dict_load=True,all_tensors_finite=True,model_forward_executed=False,
    internal_holdout_evaluated=False,test_split_accessed=False,state_hashes=hashes)))
''',timeout=60)
assert result['verified'] and result['checkpoint_sha256']==digest(HERE/'results/heads.pt')
write_json(HERE/'checkpoint_verified.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='state_hashes'}))

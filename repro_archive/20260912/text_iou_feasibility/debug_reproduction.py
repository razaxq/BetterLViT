"""Bounded read-only comparison of saved D inference under grad flag controls."""
import json
from pathlib import Path
from control import remote
from analysis import write_json
HERE=Path(__file__).resolve().parent
result=remote('''
import os,json,sys
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
sys.path[:0]=['/root/text_optimizer_d_c7080ea8','/root/BetterLViT-recipe-r2']
import numpy as np,torch
from heads import ResidualHead,prediction
from analysis import metrics
from pathlib import Path
torch.set_num_threads(4);torch.manual_seed(1219);torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
B=Path('/root/text_head_b_49905dbb');D=Path('/root/text_optimizer_d_c7080ea8')
rows=json.loads((B/'split.json').read_text())['records'];cache=json.loads((B/'results/cache_manifest.json').read_text())
rs=json.loads((B/'results/train_diagnostics.json').read_text())[0]['records'];ix=[r['index'] for r in rs[:16]]
assert all(rows[i]['eligible'] and rows[i]['partition']=='fit' for i in ix)
arrays={k:np.memmap(B/'cache'/(k+'.bin'),dtype=dtype,mode='r',shape=tuple(shape)) for k,(shape,dtype) in cache['shapes'].items()}
text=np.load(B/'cache/text.npz');emb=torch.from_numpy(text['embeddings']).cuda();masks=torch.from_numpy(text['masks']).cuda()
lookup={t:i for i,t in enumerate(cache['unique_texts'])}
f=torch.from_numpy(np.array(arrays['features'][ix])).cuda().float();z=torch.from_numpy(np.array(arrays['logits'][ix])).cuda()
y=np.unpackbits(np.array(arrays['masks'][ix]),axis=1,count=224*224,bitorder='little').reshape(-1,1,224,224)
c=[lookup[rows[i]['canonical']] for i in ix];r=[lookup[rows[i]['reference']] for i in ix]
inputs=(f,emb[c],masks[c],emb[r],masks[r],torch.ones(16,device='cuda',dtype=torch.bool))
ck=torch.load(D/'results/heads.pt',map_location='cpu',weights_only=False)
expected=json.loads((D/'results/train_diagnostics.json').read_text())[-1]['cases']['adamw/T1']['records'][:16]
results={}
for flag in (True,False):
    head=ResidualHead().cuda();head.load_state_dict(ck['heads']['adamw/T1']);head.requires_grad_(flag)
    with torch.no_grad():p=prediction(z,head(*inputs,'T1'),'T1').cpu().numpy()
    observed=[metrics(p[j],y[j]) for j in range(16)]
    results[str(flag)]=[dict(index=i,different={k:dict(got=observed[j][k],expected=e['output'][k],diff=observed[j][k]-e['output'][k])
        for k in observed[j] if observed[j][k]!=e['output'][k]}) for j,(i,e) in enumerate(zip(ix,expected))]
print(json.dumps(results))
''',timeout=120)
write_json(HERE/'reproduction_debug.json',result)
print(json.dumps(result))

"""Read checkpoint metadata only for named historical Last copies with siblings."""
from remote_ops import remote,save
import json
result=remote('''
import json,torch
from pathlib import Path
labels=['BetterLViT-dual-grain-p11','BetterLViT-paper-c0-tversky','BetterLViT-paper-p1-tcsrv21',
        'BetterLViT-paper-p2-tcsrv22','BetterLViT-paper-p3-tcsrv23','BetterLViT-paper-p4-tcsrv24']
items=[]
for label in labels:
    repo=Path('/root/autodl-tmp')/label
    matches=list((repo/'Covid19').glob('**/last_model-BetterLViT.pth.tar'))
    assert len(matches)==1
    p=matches[0];b=p.with_name('best_model-BetterLViT.pth.tar')
    assert p.is_file() and b.is_file() and not p.is_symlink() and not b.is_symlink()
    ck=torch.load(p,map_location='cpu',weights_only=True)
    items.append(dict(label=label,last=str(p),best=str(b),bytes=p.stat().st_size,
        source_git_commit=ck.get('source_git_commit'),epoch=ck.get('epoch'),epochs=ck.get('epochs'),
        history_rows=len(ck.get('epoch_history',[]))))
    del ck
print(json.dumps(items))
''')
save('storage_candidates.json',result)
print(json.dumps(result))

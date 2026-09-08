"""Read-only Train/Val data audit; no model, GPU, Test, or training inspection."""
import hashlib
import json
import subprocess
from pathlib import Path

REMOTE = r'''
import collections, hashlib, json, time
from pathlib import Path
import cv2
import numpy as np
base=Path('/root/autodl-tmp/datasets/Covid19')
result={'test_split_accessed':False,'model_loaded':False,'training_started':False,
        'audit':'native shapes and categorical-mask resize agreement','started_unix':time.time(),
        'opencv':cv2.__version__,'numpy':np.__version__,'splits':{}}
for split in ('Train_Folder','Val_Folder'):
    folder=base/split
    paths=sorted((folder/'labelcol').glob('*.png'))
    assert paths
    selected=paths if split=='Val_Folder' else [paths[i] for i in np.linspace(0,len(paths)-1,32,dtype=int)]
    shapes=collections.Counter(); values=collections.Counter(); rows=[]
    image_shapes=collections.Counter(); mask_hash=hashlib.sha256()
    for index,path in enumerate(selected):
        raw=path.read_bytes(); mask_hash.update(path.name.encode()); mask_hash.update(raw)
        mask=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
        assert mask is not None
        shapes[str(mask.shape)]+=1
        values[','.join(map(str,np.unique(mask).tolist()))]+=1
        if index<32:
            im=cv2.imread(str(folder/'img'/path.name.replace('mask_','')))
            assert im is not None
            image_shapes[str(im.shape)]+=1
        legacy=cv2.resize(mask,(224,224),interpolation=cv2.INTER_LINEAR)>0
        row={'name':path.name,'native_shape':list(mask.shape),'legacy_area':int(legacy.sum())}
        for label,flag in [('nearest',cv2.INTER_NEAREST),('nearest_exact',cv2.INTER_NEAREST_EXACT)]:
            other=cv2.resize(mask,(224,224),interpolation=flag)>0
            union=(legacy|other).sum()
            row[label]={'area':int(other.sum()),'agreement_iou':float((legacy&other).sum()/union) if union else 1.,
                'legacy_only_pixels':int((legacy&~other).sum()),'other_only_pixels':int((other&~legacy).sum())}
        rows.append(row)
    result['splits'][split]={'available_masks':len(paths),'audited_masks':len(selected),
        'mask_shapes':dict(shapes),'mask_unique_value_sets':dict(values),
        'sampled_image_shapes':dict(image_shapes),'mask_bytes_and_names_sha256':mask_hash.hexdigest(),
        'summary':{key:{'changed_images':sum(r[key]['legacy_only_pixels']+r[key]['other_only_pixels']>0 for r in rows),
            'mean_agreement_iou':float(np.mean([r[key]['agreement_iou'] for r in rows])),
            'legacy_total_area':sum(r['legacy_area'] for r in rows),
            'other_total_area':sum(r[key]['area'] for r in rows)} for key in ('nearest','nearest_exact')},
        'records':rows}
result['completed_unix']=time.time()
print(json.dumps(result))
'''

out = Path(__file__).resolve().parent / 'data_contract_audit.json'
assert not out.exists(), 'Use the saved audit; do not repeat the server read'
process = subprocess.run(['ssh','-i','C:/Users/dtftn/.ssh/seetacloud_betterlvit_ed25519','-p','21465',
    '-o','BatchMode=yes','-o','ConnectTimeout=15','root@connect.westb.seetacloud.com',
    '/root/autodl-tmp/envs/betterlvit-paper/bin/python -'], input=REMOTE,text=True,
    capture_output=True,timeout=60)
if process.returncode:
    raise RuntimeError(process.stderr)
result = json.loads(process.stdout)
result['audit_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
result['frozen_source_reference'] = 'add4908a0d6f702b0a10c4581725b535543829b8'
result['source_file_sha256'] = {name:hashlib.sha256((Path('D:/BetterLViT/race_pe_c4_work')/name).read_bytes()).hexdigest()
    for name in ('Load_Dataset.py','train_model.py','Config.py')}
out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({**{k:v for k,v in result.items() if k!='splits'},
    'splits':{k:{a:b for a,b in v.items() if a!='records'} for k,v in result['splits'].items()}},indent=2))

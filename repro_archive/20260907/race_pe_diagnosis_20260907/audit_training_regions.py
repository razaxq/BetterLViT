import sys,os,json
from pathlib import Path
sys.path[:0]=['/root/BetterLViT-race-pe-p9','/root/BetterLViT-race-pe-p9/tools']
os.environ.update(BETTERLVIT_EXPERIMENT='p9_race_pe',HF_HOME='/root/autodl-fs/betterlvit_5090_migration/root_cache/huggingface',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import torch
from torch.utils.data import DataLoader
import Config as config
from Load_Dataset import ImageToImage2D,ValGenerator
from utils import read_text
text=read_text(Path(config.task_dataset)/'Train_Val_text.xlsx')
ds=ImageToImage2D(config.train_dataset,config.task_name,text,ValGenerator([224,224]),image_size=224)
positive=conflicts=total=0;examples=[]
for batch,names in DataLoader(ds,batch_size=32,num_workers=0):
 basis=batch['race_zone_basis']; mask=batch['label'].float().unsqueeze(1)
 occ=(mask*basis).sum((2,3))/basis.sum((2,3)).clamp_min(1)
 known=batch['race_slot_targets'][:,:6]==1; bad=known&(occ==0)
 positive+=int(known.sum());conflicts+=int(bad.sum());total+=len(names)
 for i in range(len(names)):
  if bad[i].any() and len(examples)<10: examples.append({'image':names[i],'report':text['mask_'+names[i]],'positive_slots':torch.where(known[i])[0].tolist(),'empty_positive_slots':torch.where(bad[i])[0].tolist(),'occupancy':occ[i].tolist()})
r={'split':'train','augmentation':'none; pre-augmentation geometry','samples':total,'positive_regions':positive,'text_positive_mask_empty':conflicts,'conflict_fraction':conflicts/positive,'test_split_accessed':False,'examples':examples}
Path('/root/race_pe_diagnosis_20260907/training_regions.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))

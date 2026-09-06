"""Small text-head-only learning probe; no segmentation optimization or Test."""
import os
os.environ.setdefault('BETTERLVIT_EXPERIMENT','p10_race_pe_v2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,hashlib,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModel
import Config as config
from Load_Dataset import _build_tokenizer
from utils import read_text
from race_semantics import parse_report_mentions
from nets.race_pe import RACEPEV2


def score(logits,labels):
    prob=logits.sigmoid(); pred=prob>.5; known=labels>=0; positive=labels==1
    tp=(pred&positive&known).sum(0).float(); fp=(pred&~positive&known).sum(0).float()
    fn=(~pred&positive&known).sum(0).float(); tn=(~pred&~positive&known).sum(0).float()
    f1=2*tp/(2*tp+fp+fn).clamp_min(1)
    bce=F.binary_cross_entropy_with_logits(logits,labels.clamp_min(0),reduction='none')
    return dict(macro_f1=float(f1.mean()),macro_specificity=float((tn/(tn+fp).clamp_min(1)).mean()),
        masked_bce=float((bce*known).sum()/known.sum().clamp_min(1)),
        mean_probability=float(prob.mean()),fraction_above_099=float((prob>.99).float().mean()),
        negative_labels=int(((labels==0)&known).sum()),positive_labels=int(positive.sum()))


def main():
    random.seed(1219); np.random.seed(1219);torch.manual_seed(1219)
    torch.use_deterministic_algorithms(True)
    out=Path(os.environ['RACE_PROBE_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
    text=read_text(Path(config.task_dataset)/'Train_Val_text.xlsx')
    def rows(path):return [text[p.name] for p in sorted((Path(path)/'labelcol').iterdir())]
    train_raw,val_raw=rows(config.train_dataset),rows(config.val_dataset)
    tok=_build_tokenizer()
    unique=sorted(set(train_raw+val_raw))
    tokens=tok(unique,padding='max_length',truncation=True,max_length=32,return_tensors='pt')
    visible=tok.batch_decode(tokens['input_ids'],skip_special_tokens=True)
    y=torch.stack([parse_report_mentions(t)[:6] for t in visible])
    idx={s:i for i,s in enumerate(unique)}
    train_ids=torch.tensor([idx[t] for t in train_raw]);val_ids=torch.tensor([idx[t] for t in val_raw])
    groups=sorted(set(visible[i] for i in train_ids.tolist()))
    random.Random(1219).shuffle(groups);hold=set(groups[:max(1,len(groups)//5)])
    fit_ids=torch.tensor([i for i in train_ids.tolist() if visible[i] not in hold])
    hold_ids=torch.tensor(sorted(set(i for i in train_ids.tolist() if visible[i] in hold)))
    encoder=AutoModel.from_pretrained(config.text_encoder_name,trust_remote_code=True).cuda().eval()
    features=[]
    with torch.inference_mode():
        for start in range(0,len(unique),32):
            ids=tokens['input_ids'][start:start+32].cuda();mask=tokens['attention_mask'][start:start+32].cuda()
            h=encoder(input_ids=ids,attention_mask=mask).last_hidden_state
            features.append(((h*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)).cpu())
    x=torch.cat(features);del encoder;torch.cuda.empty_cache()
    torch.manual_seed(1219)
    head=RACEPEV2(channels=(8,),hidden_channels=8).slot_head.cuda()
    opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
    x=x.cuda();y=y.cuda();history=[]
    for epoch in range(40):
        head.train()
        order=fit_ids[torch.randperm(len(fit_ids))]
        for start in range(0,len(order),128):
            indices=order[start:start+128];logits=head(x[indices])[:,:6];labels=y[indices]
            known=labels>=0
            loss=(F.binary_cross_entropy_with_logits(logits,labels.clamp_min(0),reduction='none')*known).sum()/known.sum().clamp_min(1)
            opt.zero_grad();loss.backward();opt.step()
        if epoch in (0,4,9,19,39):
            head.eval()
            with torch.no_grad():
                row=dict(epoch=epoch+1,heldout=score(head(x[hold_ids])[:,:6],y[hold_ids]),
                         validation=score(head(x[val_ids])[:,:6],y[val_ids]))
            history.append(row);print(json.dumps(row),flush=True)
    with torch.no_grad():
        logits=head(x[val_ids])[:,:6]; normal=score(logits,y[val_ids])
        shuffled=score(logits.roll(1,0),y[val_ids]);allone=score(torch.full_like(logits,10),y[val_ids])
        heldout=score(head(x[hold_ids])[:,:6],y[hold_ids])
    gate=(normal['macro_f1']>=.85 and heldout['macro_f1']>=.85 and normal['macro_specificity']>=.9
          and normal['macro_f1']-shuffled['macro_f1']>=.15 and normal['masked_bce']<allone['masked_bce'])
    result=dict(status='complete',segmentation_training_performed=False,text_head_training_performed=True,
        test_split_accessed=False,seed=1219,epochs=40,lr=.001,unique_raw_reports=len(unique),
        train_visible_groups=len(groups),heldout_visible_groups=len(hold),fit_images=len(fit_ids),
        validation_images=len(val_ids),head_source='random initialization, frozen CXR-BERT masked mean',
        target='explicit positive region mention in encoder-visible text, not lesion presence',
        checkpoint_selection='fixed epoch40; no validation selection',validation=normal,heldout_templates=heldout,
        shuffled_validation=shuffled,all_positive_validation=allone,passes_learnability_gate=gate,history=history)
    (out/'mention_probe.json').write_text(json.dumps(result,indent=2))
    # Diagnostic-only head. Do not silently initialize a formal segmentation run from it.
    torch.save(head.cpu().state_dict(),out/'diagnostic_only_mention_head.pth')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
